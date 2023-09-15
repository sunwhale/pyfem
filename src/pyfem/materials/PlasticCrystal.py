# -*- coding: utf-8 -*-
"""

"""
from copy import deepcopy

import numpy as np
from numpy import zeros, ndarray, sqrt, sign, dot, array, einsum, eye, ones, maximum, abs, transpose, all, delete
from numpy.linalg import solve, inv

from pyfem.fem.Timer import Timer
from pyfem.fem.constants import DTYPE
from pyfem.io.Material import Material
from pyfem.io.Section import Section
from pyfem.materials.BaseMaterial import BaseMaterial
from pyfem.materials.crystal_slip_system import generate_mn
from pyfem.materials.ElasticIsotropic import get_stiffness_from_young_poisson
from pyfem.utils.colors import error_style
from pyfem.utils.mechanics import get_transformation, get_voigt_transformation


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
        'tolerance': ('float', '判断屈服的误差容限'),
        'total_number_of_slips': ('int', '总的滑移系数量'),
        'elastic': ('dict', '弹性参数字典'),
        'C': ('ndarray', '弹性矩阵'),
        'slip_system': ('str', ''),
        'K': ('ndarray', '硬化系数矩阵'),
        'a': ('ndarray', '硬化系数矩阵'),
        'q': ('ndarray', '硬化系数矩阵'),
        'theta': ('ndarray', '硬化系数矩阵'),
        'c1': ('ndarray', '硬化系数矩阵'),
        'c2': ('ndarray', '硬化系数矩阵'),
        'r0': ('ndarray', '硬化系数矩阵'),
        'b': ('ndarray', '硬化系数矩阵'),
        'Q': ('ndarray', '硬化系数矩阵'),
        'H': ('ndarray', '硬化系数矩阵'),
        'u_global': ('ndarray', '全局坐标系下的1号矢量'),
        'v_global': ('ndarray', '全局坐标系下的2号矢量'),
        'w_global': ('ndarray', '全局坐标系下的3号矢量'),
        'u_grain': ('ndarray', '晶粒坐标系下的1号矢量'),
        'v_grain': ('ndarray', '晶粒坐标系下的2号矢量'),
        'w_grain': ('ndarray', '晶粒坐标系下的3号矢量'),
        'T': ('ndarray', '硬化系数矩阵'),
        'T_vogit': ('ndarray', '硬化系数矩阵'),
        'm': ('ndarray', '硬化系数矩阵'),
        'n': ('ndarray', '硬化系数矩阵'),
        'MAX_NITER': ('ndarray', '硬化系数矩阵'),
    }

    __slots__ = BaseMaterial.__slots__ + [slot for slot in __slots_dict__.keys()]

    def __init__(self, material: Material, dimension: int, section: Section) -> None:
        super().__init__(material, dimension, section)
        self.allowed_section_types = ('Volume', 'PlaneStress', 'PlaneStrain')

        self.data_keys = []

        if len(self.material.data) != len(self.data_keys):
            raise NotImplementedError(error_style(self.get_data_length_error_msg()))
        else:
            for i, key in enumerate(self.data_keys):
                self.data_dict[key] = material.data[i]

        self.tolerance: float = 1.0e-6
        self.MAX_NITER = 8
        self.theta: float = 0.5
        self.total_number_of_slips: int = material.data_dict['total_number_of_slips']
        self.slip_system = material.data_dict['slip_system']

        self.elastic = material.data_dict['elastic']
        self.C = self.create_elastic_stiffness(self.elastic)

        self.K = material.data_dict['K']
        self.a = material.data_dict['a']
        self.q = material.data_dict['q']

        self.theta = material.data_dict['theta']
        self.c1 = material.data_dict['c1']
        self.c2 = material.data_dict['c2']
        self.r0 = material.data_dict['r0']
        self.b = material.data_dict['b']
        self.Q = material.data_dict['Q']

        self.H = ones(shape=(self.total_number_of_slips, self.total_number_of_slips), dtype=DTYPE)

        self.u_global = array(section.data_dict['u_global'])
        self.v_global = array(section.data_dict['v_global'])
        self.w_global = array(section.data_dict['w_global'])

        self.u_grain = array(section.data_dict['u_grain'])
        self.v_grain = array(section.data_dict['v_grain'])
        self.w_grain = array(section.data_dict['w_grain'])

        self.T = get_transformation(self.u_grain, self.v_grain, self.w_grain, self.u_global, self.v_global,
                                    self.w_global)
        self.T_vogit = get_voigt_transformation(self.T)

        _, self.m, self.n = generate_mn('slip', self.slip_system, 1.0)

        self.m = dot(self.m, self.T)
        self.n = dot(self.n, self.T)
        self.C = dot(dot(self.T_vogit, self.C), transpose(self.T_vogit))
        self.create_tangent()

    def create_tangent(self):
        if self.section.type in self.allowed_section_types:
            pass
        else:
            raise NotImplementedError(error_style(self.get_section_type_error_msg()))

    def create_elastic_stiffness(self, elastic: dict):
        symmetry = elastic['symmetry']
        if symmetry == 'isotropic':
            C11 = elastic['C11']
            C12 = elastic['C12']
            C44 = elastic['C44']
            C = array([[C11, C12, C12, 0, 0, 0],
                       [C12, C11, C12, 0, 0, 0],
                       [C12, C12, C11, 0, 0, 0],
                       [0, 0, 0, C44, 0, 0],
                       [0, 0, 0, 0, C44, 0],
                       [0, 0, 0, 0, 0, C44]], dtype=DTYPE)
        else:
            raise NotImplementedError(
                error_style(f'the symmetry type \"{symmetry}\" of elastic stiffness is not supported'))
        return C

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

        if self.section.type == 'PlaneStrain':
            strain = array([strain[0], strain[1], 0.0, strain[2], 0.0, 0.0])
            dstrain = array([dstrain[0], dstrain[1], 0.0, dstrain[2], 0.0, 0.0])

        np.set_printoptions(precision=12, linewidth=256, suppress=True)

        K = self.K
        a = self.a
        q = self.q
        dt = timer.dtime
        theta = self.theta
        c1 = self.c1
        c2 = self.c2
        r0 = self.r0
        b = self.b
        Q = self.Q
        H = self.H
        C = self.C
        m = self.m
        n = self.n

        if state_variable == {} or timer.time0 == 0.0:
            state_variable['m_e'] = m
            state_variable['n_e'] = n
            mxn = transpose(array([m[:, 0] * n[:, 0],
                                   m[:, 1] * n[:, 1],
                                   m[:, 2] * n[:, 2],
                                   2.0 * m[:, 0] * n[:, 1],
                                   2.0 * m[:, 0] * n[:, 2],
                                   2.0 * m[:, 1] * n[:, 2]]))
            nxm = transpose(array([n[:, 0] * m[:, 0],
                                   n[:, 1] * m[:, 1],
                                   n[:, 2] * m[:, 2],
                                   2.0 * n[:, 0] * m[:, 1],
                                   2.0 * n[:, 0] * m[:, 2],
                                   2.0 * n[:, 1] * m[:, 2]]))
            P = 0.5 * (mxn + nxm)
            state_variable['stress'] = zeros(shape=6, dtype=DTYPE)
            state_variable['tau'] = dot(P, state_variable['stress'])
            state_variable['gamma'] = zeros(shape=self.total_number_of_slips, dtype=DTYPE)
            state_variable['rho'] = zeros(shape=self.total_number_of_slips, dtype=DTYPE)
            state_variable['alpha'] = zeros(shape=self.total_number_of_slips, dtype=DTYPE)
            state_variable['r'] = zeros(shape=self.total_number_of_slips, dtype=DTYPE) + r0

        rho = deepcopy(state_variable['rho'])
        m_e = deepcopy(state_variable['m_e'])
        n_e = deepcopy(state_variable['n_e'])
        gamma = deepcopy(state_variable['gamma'])
        stress = deepcopy(state_variable['stress'])
        tau = deepcopy(state_variable['tau'])
        alpha = deepcopy(state_variable['alpha'])
        r = deepcopy(state_variable['r'])

        delta_gamma = zeros(shape=self.total_number_of_slips, dtype=DTYPE)
        delta_stress = zeros(shape=6, dtype=DTYPE)
        delta_tau = zeros(shape=self.total_number_of_slips, dtype=DTYPE)
        delta_alpha = zeros(shape=self.total_number_of_slips, dtype=DTYPE)
        delta_r = zeros(shape=self.total_number_of_slips, dtype=DTYPE)

        is_convergence = False

        for niter in range(self.MAX_NITER):
            m_exn_e = transpose(array([m_e[:, 0] * n_e[:, 0],
                                       m_e[:, 1] * n_e[:, 1],
                                       m_e[:, 2] * n_e[:, 2],
                                       2.0 * m_e[:, 0] * n_e[:, 1],
                                       2.0 * m_e[:, 0] * n_e[:, 2],
                                       2.0 * m_e[:, 1] * n_e[:, 2]]))

            n_exm_e = transpose(array([n_e[:, 0] * m_e[:, 0],
                                       n_e[:, 1] * m_e[:, 1],
                                       n_e[:, 2] * m_e[:, 2],
                                       2.0 * n_e[:, 0] * m_e[:, 1],
                                       2.0 * n_e[:, 0] * m_e[:, 2],
                                       2.0 * n_e[:, 1] * m_e[:, 2]]))

            P = 0.5 * (m_exn_e + n_exm_e)
            Omega = 0.5 * (m_exn_e - n_exm_e)
            Omega[:, 3:] *= 0.5

            # S = dot(P, C) + Omega * stress - stress * Omega
            S = dot(P, C)

            X = (abs(tau - alpha) - r) / K
            gamma_dot = a * maximum(X, 0.0) ** q * sign(tau - alpha)

            if niter == 0:
                gamma_dot_init = deepcopy(gamma_dot)

            term1 = dt * theta
            term2 = term1 * a * q * maximum(X, 0.0) ** (q - 1.0) / K
            term3 = term1 * maximum(X, 0) * a * q * maximum(X, 0.0) ** (q - 1.0) / K
            term4 = einsum('ik, jk->ij', S, P)

            A = eye(self.total_number_of_slips, dtype=DTYPE)
            A += term2 * term4
            # A += H * term3 * sign(gamma_dot) * sign(tau - alpha)
            A += term2 * (c1 - c2 * alpha * sign(gamma_dot)) * eye(self.total_number_of_slips, dtype=DTYPE)
            A += term2 * b * Q * H * (1.0 - b * rho) * sign(tau - alpha) * sign(gamma_dot)

            if niter == 0:
                rhs = dt * gamma_dot + term2 * dot(S, dstrain)
            else:
                rhs = term1 * (gamma_dot - gamma_dot_init) + gamma_dot_init * dt - delta_gamma

            d_delta_gamma = solve(transpose(A), rhs)
            delta_gamma += d_delta_gamma

            delta_elastic_strain = dstrain - dot(delta_gamma, P)
            delta_tau = dot(S, delta_elastic_strain)
            delta_stress = dot(C, delta_elastic_strain)
            delta_alpha = c1 * delta_gamma - c2 * abs(delta_gamma) * alpha
            delta_rho = (1.0 - b * rho) * abs(delta_gamma)
            delta_r = b * Q * dot(H, delta_rho)

            gamma = deepcopy(state_variable['gamma']) + delta_gamma
            tau = deepcopy(state_variable['tau']) + delta_tau
            stress = deepcopy(state_variable['stress']) + delta_stress
            alpha = deepcopy(state_variable['alpha']) + delta_alpha
            rho = deepcopy(state_variable['rho']) + delta_rho
            r = deepcopy(state_variable['r']) + delta_r

            X = (abs(tau - alpha) - r) / K
            gamma_dot = a * maximum(X, 0.0) ** q * sign(tau - alpha)
            residual = dt * theta * gamma_dot + dt * (1.0 - theta) * gamma_dot_init - delta_gamma

            # if element_id == 0 and iqp == 0:
            #     print('residual', residual)

            if all(residual < self.tolerance):
                is_convergence = True
                break

        ddgdde = term2.reshape((self.total_number_of_slips, 1)) * S
        ddgdde = dot(inv(A), ddgdde)
        ddsdde = C - einsum('ki, kj->ij', S, ddgdde)

        state_variable_new['m_e'] = m_e
        state_variable_new['n_e'] = n_e
        state_variable_new['stress'] = stress
        state_variable_new['gamma'] = gamma
        state_variable_new['tau'] = tau
        state_variable_new['alpha'] = alpha
        state_variable_new['r'] = r
        state_variable_new['rho'] = rho

        some_energy = 0.5 * sum(strain * stress)

        if self.section.type == 'PlaneStrain':
            ddsdde = delete(delete(ddsdde, [2, 4, 5], axis=0), [2, 4, 5], axis=1)
            stress = delete(stress, [2, 4, 5])

        output = {'stress': stress, 'plastic_energy': some_energy}

        return ddsdde, output


if __name__ == "__main__":
    # from pyfem.utils.visualization import print_slots_dict
    #
    # print_slots_dict(PlasticCrystal.__slots_dict__)

    from pyfem.Job import Job

    job = Job(r'..\..\..\examples\mechanical\1element\hex20_crystal\Job-1.toml')

    job.props.amplitudes[0].show()

    job.run()
