# -*- coding: utf-8 -*-
"""

"""
from copy import deepcopy

import numpy as np
from numpy import zeros, ndarray, sqrt, sign, dot, array, einsum, eye, ones, maximum, abs
from numpy.linalg import solve, inv

from pyfem.fem.Timer import Timer
from pyfem.fem.constants import DTYPE
from pyfem.io.Material import Material
from pyfem.io.Section import Section
from pyfem.materials.BaseMaterial import BaseMaterial
from pyfem.materials.ElasticIsotropic import get_stiffness_from_young_poisson
from pyfem.utils.colors import error_style


class PlasticCrystal(BaseMaterial):
    r"""
    晶体塑性材料。

    支持的截面属性：('Volume', 'PlaneStress', 'PlaneStrain')

    :ivar E: Young's modulus E
    :vartype E: float

    :ivar nu: Poisson's ratio nu
    :vartype nu: float

    :ivar yield_stress: Yield stress
    :vartype yield_stress: float

    :ivar hard: Hardening coefficient
    :vartype hard: float

    :ivar EBULK3: 3倍体积模量
    :vartype EBULK3: float

    :ivar EG: 剪切模量
    :vartype EG: float

    :ivar EG2: 2倍剪切模量
    :vartype EG2: float

    :ivar EG3: 3倍剪切模量
    :vartype EG3: float

    :ivar ELAM: 拉梅常数
    :vartype ELAM: float

    :ivar tolerance: 判断屈服的误差容限
    :vartype tolerance: float

    :ivar total_number_of_slips: 总的滑移系数量
    :vartype total_number_of_slips: int

    :ivar h_matrix: 硬化系数矩阵
    :vartype h_matrix: ndarray

    """

    __slots_dict__: dict = {
        'E': ('float', 'Young\'s modulus E'),
        'nu': ('float', 'Poisson\'s ratio nu'),
        'yield_stress': ('float', 'Yield stress'),
        'hard': ('float', 'Hardening coefficient'),
        'EBULK3': ('float', '3倍体积模量'),
        'EG': ('float', '剪切模量'),
        'EG2': ('float', '2倍剪切模量'),
        'EG3': ('float', '3倍剪切模量'),
        'ELAM': ('float', '拉梅常数'),
        'tolerance': ('float', '判断屈服的误差容限'),
        'total_number_of_slips': ('int', '总的滑移系数量'),
        'h_matrix': ('ndarray', '硬化系数矩阵')
    }

    __slots__ = BaseMaterial.__slots__ + [slot for slot in __slots_dict__.keys()]

    def __init__(self, material: Material, dimension: int, section: Section) -> None:
        super().__init__(material, dimension, section)
        self.allowed_section_types = ('Volume', 'PlaneStress', 'PlaneStrain')

        self.data_keys = ['Young\'s modulus E', 'Poisson\'s ratio nu', 'Yield stress', 'Hardening coefficient']

        if len(self.material.data) != len(self.data_keys):
            raise NotImplementedError(error_style(self.get_data_length_error_msg()))
        else:
            for i, key in enumerate(self.data_keys):
                self.data_dict[key] = material.data[i]

        self.E: float = self.data_dict['Young\'s modulus E']
        self.nu: float = self.data_dict['Poisson\'s ratio nu']
        self.yield_stress: float = self.data_dict['Yield stress']
        self.hard: float = self.data_dict['Hardening coefficient']

        self.EBULK3: float = self.E / (1.0 - 2.0 * self.nu)
        self.EG2: float = self.E / (1.0 + self.nu)
        self.EG: float = self.EG2 / 2.0
        self.EG3: float = 3.0 * self.EG
        self.ELAM: float = (self.EBULK3 - self.EG2) / 3.0
        self.tolerance: float = 1.0e-10

        self.total_number_of_slips: int = 12
        self.h_matrix: ndarray = ones((self.total_number_of_slips, self.total_number_of_slips))

        self.create_tangent()

    def create_tangent(self):
        if self.section.type in self.allowed_section_types:
            self.tangent = get_stiffness_from_young_poisson(self.dimension, self.E, self.nu, self.section.type)
        else:
            raise NotImplementedError(error_style(self.get_section_type_error_msg()))

    def get_tangent(self, variable: dict[str, ndarray],
                    state_variable: dict[str, ndarray],
                    state_variable_new: dict[str, ndarray],
                    element_id: int,
                    iqp: int,
                    ntens: int,
                    ndi: int,
                    nshr: int,
                    timer: Timer) -> tuple[ndarray, dict[str, ndarray]]:

        strain = variable['strain']
        dstrain = variable['dstrain']

        np.set_printoptions(precision=10, linewidth=256, suppress=True)

        C11 = 169727.0
        C12 = 104026.0
        C44 = 86000.0
        g = 240.0
        a = 0.00025
        q = 3
        dt = timer.dtime
        theta = 0.5
        c1 = 2000.0
        c2 = 10.0
        r0 = 10.0
        b = 1.0
        Q = 20.0
        H = ones(shape=(self.total_number_of_slips, self.total_number_of_slips), dtype=DTYPE)

        if state_variable == {}:
            state_variable['rho_m'] = zeros(shape=self.total_number_of_slips, dtype=DTYPE)
            state_variable['rho_di'] = zeros(shape=self.total_number_of_slips, dtype=DTYPE)
            state_variable['rho'] = zeros(shape=self.total_number_of_slips, dtype=DTYPE)
            state_variable['m_e'] = zeros(shape=(self.total_number_of_slips, 3), dtype=DTYPE)
            state_variable['n_e'] = zeros(shape=(self.total_number_of_slips, 3), dtype=DTYPE)
            state_variable['gamma'] = zeros(shape=self.total_number_of_slips, dtype=DTYPE)
            state_variable['stress'] = zeros(shape=6, dtype=DTYPE)
            state_variable['tau'] = zeros(shape=self.total_number_of_slips, dtype=DTYPE)
            state_variable['alpha'] = zeros(shape=self.total_number_of_slips, dtype=DTYPE)
            state_variable['r'] = zeros(shape=self.total_number_of_slips, dtype=DTYPE) + r0

        rho_m = deepcopy(state_variable['rho_m'])
        rho_di = deepcopy(state_variable['rho_di'])
        rho = deepcopy(state_variable['rho'])
        m_e = deepcopy(state_variable['m_e'])
        n_e = deepcopy(state_variable['n_e'])
        gamma = deepcopy(state_variable['gamma'])
        stress = deepcopy(state_variable['stress'])
        tau = deepcopy(state_variable['tau'])
        alpha = deepcopy(state_variable['alpha'])
        r = deepcopy(state_variable['r'])

        C = array([[C11, C12, C12, 0, 0, 0],
                   [C12, C11, C12, 0, 0, 0],
                   [C12, C12, C11, 0, 0, 0],
                   [0, 0, 0, C44, 0, 0],
                   [0, 0, 0, 0, C44, 0],
                   [0, 0, 0, 0, 0, C44]], dtype=DTYPE)

        m = array([[0.000000, -0.707107, 0.707107],
                   [0.707107, 0.000000, -0.707107],
                   [-0.707107, 0.707107, 0.000000],
                   [0.707107, 0.000000, 0.707107],
                   [0.707107, 0.707107, 0.000000],
                   [0.000000, -0.707107, 0.707107],
                   [0.000000, 0.707107, 0.707107],
                   [0.707107, 0.707107, 0.000000],
                   [0.707107, 0.000000, -0.707107],
                   [0.000000, 0.707107, 0.707107],
                   [0.707107, 0.000000, 0.707107],
                   [-0.707107, 0.707107, 0.000000]], dtype=DTYPE)

        n = array([[0.577350, 0.577350, 0.577350],
                   [0.577350, 0.577350, 0.577350],
                   [0.577350, 0.577350, 0.577350],
                   [-0.577350, 0.577350, 0.577350],
                   [-0.577350, 0.577350, 0.577350],
                   [-0.577350, 0.577350, 0.577350],
                   [0.577350, -0.577350, 0.577350],
                   [0.577350, -0.577350, 0.577350],
                   [0.577350, -0.577350, 0.577350],
                   [0.577350, 0.577350, -0.577350],
                   [0.577350, 0.577350, -0.577350],
                   [0.577350, 0.577350, -0.577350]], dtype=DTYPE)

        mxn = array([m[:, 0] * n[:, 0],
                     m[:, 1] * n[:, 1],
                     m[:, 2] * n[:, 2],
                     2.0 * m[:, 0] * n[:, 1],
                     2.0 * m[:, 0] * n[:, 2],
                     2.0 * m[:, 1] * n[:, 2]]).T

        nxm = array([n[:, 0] * m[:, 0],
                     n[:, 1] * m[:, 1],
                     n[:, 2] * m[:, 2],
                     2.0 * n[:, 0] * m[:, 1],
                     2.0 * n[:, 0] * m[:, 2],
                     2.0 * n[:, 1] * m[:, 2]]).T

        P = 0.5 * (mxn + nxm)
        Omega = 0.5 * (mxn - nxm)
        Omega[:, 3:] *= 0.5

        if timer.time0 == 0:
            tau = dot(P, stress)

        # S = dot(P, C) + Omega * stress - stress * Omega
        S = dot(P, C)

        delta_gamma = zeros(shape=self.total_number_of_slips, dtype=DTYPE)
        delta_stress = zeros(shape=6, dtype=DTYPE)
        delta_tau = zeros(shape=self.total_number_of_slips, dtype=DTYPE)
        delta_alpha = zeros(shape=self.total_number_of_slips, dtype=DTYPE)
        delta_r = zeros(shape=self.total_number_of_slips, dtype=DTYPE)

        # gamma0 = array(
        #     [0.004407919226, -0.004407919226, -0.000000000000, 0.004407919226, 0.000000000000, 0.004407919226,
        #      0.004407919226, -0.000000000000, -0.004407919226, -0.004407919226, -0.004407919226, -0.000000000000])
        # tau = array([-0.0000000000, 371.1704414891, -371.1704414891, -593.5334755464, -593.5334755464, 0.0000000000,
        #        -222.3630340573, 371.1704414891, 593.5334755464, -222.3630340573, 371.1704414891, -593.5334755464])
        # alpha = array([-0.000000000000, 0.440044677445, -0.440044677445, -1.831500687385, -1.831500687385, 0.000000000000,
        #        -0.082904534703, 0.440044677445, 1.831500687385, -0.082904534703, 0.440044677445, -1.831500687385])
        # r = array([10.0922226653, 10.0922226653, 10.0922226653, 10.0922226653, 10.0922226653, 10.0922226653, 10.0922226653,
        #        10.0922226653, 10.0922226653, 10.0922226653, 10.0922226653, 10.0922226653])
        # g = 240.0046124822

        X = (abs(tau - alpha) - r) / g

        gamma_dot = a * maximum(X, 0) ** q * sign(tau - alpha)

        # dt = 0.1

        term1 = dt * theta
        term2 = term1 * a * q * maximum(X, 0) ** (q - 1) / g
        term3 = term1 * maximum(X, 0) * a * q * maximum(X, 0) ** (q - 1) / g
        term4 = einsum('ik, jk->ij', S, P)

        A = eye(self.total_number_of_slips, dtype=DTYPE)
        A += term2 * term4
        A += H * term3 * sign(gamma_dot) * sign(tau - alpha)
        A += term2 * (c1 - c2 * alpha * sign(gamma_dot)) * eye(self.total_number_of_slips, dtype=DTYPE)
        A += term2 * b * Q * H * (1 - b * rho) * sign(tau - alpha) * sign(gamma_dot)

        # dstrain = array([0.0016691705, -0.0006941055, -0.0006941055, -0.0004962442, -0.0004962442,  0.0000000000])

        rhs = dt * gamma_dot + term2 * dot(S, dstrain)

        # A0 = array([  [1.0000000000,  0.0000000000,  0.0000000000,  0.0000000000,  0.0000000000,  0.0000000000,  0.0000000000,  0.0000000000,  0.0000000000,  0.0000000000,  0.0000000000,  0.0000000000],
        #          [-0.0089121730,  1.0185511292, -0.0089273433, -0.0077337920, -0.0089273384,  0.0012011323,  0.0011859608, -0.0011859621, -0.0023795134,  0.0089121733,  0.0077337969,  0.0011859670],
        #          [-0.0089273449, -0.0089273433,  1.0185511292,  0.0089273384,  0.0077337920,  0.0011859605, -0.0089121733, -0.0077337969,  0.0011859670, -0.0011859608,  0.0011859621, -0.0023795134],
        #          [-0.0031248434, -0.0201155064,  0.0232197749,  1.0482372871,  0.0232197622,  0.0231786293,  0.0232197782,  0.0030836978, -0.0201154937,  0.0031248426,  0.0061879663, -0.0030837105],
        #           [0.0030836937, -0.0232197749,  0.0201155064,  0.0232197622,  1.0482372871, -0.0232197790,  0.0031248426,  0.0061879663,  0.0030837105,  0.0232197782,  0.0030836978,  0.0201154937],
        #           [0.0000000000,  0.0000000000,  0.0000000000,  0.0000000000,  0.0000000000,  1.0000000000,  0.0000000000,  0.0000000000,  0.0000000000,  0.0000000000,  0.0000000000,  0.0000000000],
        #           [0.0026720844,  0.0004106280, -0.0030852630,  0.0030903612,  0.0004157263,  0.0026720844,  1.0064223302,  0.0030852630, -0.0030903612,  0.0008289064,  0.0004106280,  0.0004157263],
        #           [0.0089273449, -0.0011859621, -0.0077337969,  0.0011859670,  0.0023795134, -0.0011859605,  0.0089121733,  1.0185511292,  0.0089273384,  0.0011859608,  0.0089273433, -0.0077337920],
        #           [0.0031248434, -0.0061879663,  0.0030836978, -0.0201154937,  0.0030837105, -0.0231786293, -0.0232197782,  0.0232197749,  1.0482372871, -0.0031248426,  0.0201155064, -0.0232197622],
        #          [-0.0026771855,  0.0030852630, -0.0004106280,  0.0004157263,  0.0030903612, -0.0026771855,  0.0008289064,  0.0004106280, -0.0004157263,  1.0064223302,  0.0030852630,  0.0030903612],
        #          [-0.0089121730,  0.0077337969,  0.0011859621,  0.0023795134,  0.0011859670,  0.0012011323,  0.0011859608,  0.0089273433,  0.0077337920,  0.0089121733,  1.0185511292, -0.0089273384],
        #           [0.0030836937,  0.0030836978, -0.0061879663, -0.0030837105,  0.0201154937, -0.0232197790,  0.0031248426, -0.0201155064, -0.0232197622,  0.0232197782, -0.0232197749,  1.0482372871]])
        #
        # rhs = array([0.0000000000,  0.0001010359, -0.0001010359, -0.0004299252, -0.0004299252,  0.0000000000, -0.0000215316,  0.0001010359,  0.0004299252, -0.0000215316,  0.0001010359, -0.0004299252])
        delta_gamma += solve(A, rhs)

        # print(delta_gamma)

        delta_elastic_strain = dstrain - dot(delta_gamma, P)

        delta_tau += dot(S, delta_elastic_strain)

        delta_stress = dot(C, delta_elastic_strain)

        delta_alpha = c1 * delta_gamma + c2 * abs(delta_gamma) * alpha

        delta_rho = (1.0 - b * rho) * abs(delta_gamma)

        delta_r = b * Q * dot(H, delta_rho)

        gamma = deepcopy(state_variable['gamma']) + delta_gamma

        tau = deepcopy(state_variable['tau']) + delta_tau

        stress = deepcopy(state_variable['stress']) + delta_stress

        # stress = dot(C, strain + delta_elastic_strain)

        alpha = deepcopy(state_variable['alpha']) + delta_alpha

        rho = deepcopy(state_variable['rho']) + delta_rho

        r = deepcopy(state_variable['r']) + delta_r

        ddgdde = term2.reshape((self.total_number_of_slips, 1)) * S

        ddgdde = dot(inv(A), ddgdde)

        state_variable_new['rho_m'] = rho_m
        state_variable_new['rho_di'] = rho_di
        state_variable_new['rho'] = rho_di
        state_variable_new['m_e'] = m_e
        state_variable_new['n_e'] = n_e
        state_variable_new['gamma'] = gamma
        state_variable_new['stress'] = stress
        state_variable_new['tau'] = tau
        state_variable_new['alpha'] = alpha
        state_variable_new['r'] = r

        output = {'stress': stress}
        ddsdde = C - einsum('ki, kj->ij', S, ddgdde)

        np.set_printoptions(precision=4, linewidth=256)
        print(stress)

        # if element_id == 0 and iqp == 0:
        #     print(gamma)
        #     # print(delta_elastic_strain)
        #     print(stress)

        return ddsdde, output


def get_smises(s: ndarray) -> float:
    if len(s) == 3:
        smises = sqrt(s[0] ** 2 + s[1] ** 2 - s[0] * s[1] + 3 * s[2] ** 2)
        return float(smises)
    elif len(s) >= 3:
        smises = (s[0] - s[1]) ** 2 + (s[1] - s[2]) ** 2 + (s[2] - s[0]) ** 2
        smises += 6 * sum([i ** 2 for i in s[3:]])
        smises = sqrt(0.5 * smises)
        return float(smises)
    else:
        raise NotImplementedError(error_style(f'unsupported stress dimension {len(s)}'))


if __name__ == "__main__":
    # from pyfem.utils.visualization import print_slots_dict
    #
    # print_slots_dict(PlasticCrystal.__slots_dict__)

    from pyfem.Job import Job

    job = Job(r'..\..\..\examples\mechanical\1element\hex20_crystal\Job-1.toml')

    # job.assembly.element_data_list[0].material_data_list[0].show()

    # print(job.assembly.element_data_list[0].qp_state_variables[0]['n_e'])

    job.run()
