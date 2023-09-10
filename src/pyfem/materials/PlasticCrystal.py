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
        'C11': ('ndarray', '硬化系数矩阵'),
        'C12': ('ndarray', '硬化系数矩阵'),
        'C44': ('ndarray', '硬化系数矩阵'),
        'C': ('ndarray', '硬化系数矩阵'),
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
        'u': ('ndarray', '硬化系数矩阵'),
        'v': ('ndarray', '硬化系数矩阵'),
        'w': ('ndarray', '硬化系数矩阵'),
        'u_prime': ('ndarray', '硬化系数矩阵'),
        'v_prime': ('ndarray', '硬化系数矩阵'),
        'w_prime': ('ndarray', '硬化系数矩阵'),
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

        self.data_keys = ['Young\'s modulus E', 'Poisson\'s ratio nu', 'Yield stress', 'Hardening coefficient']

        if len(self.material.data) != len(self.data_keys):
            raise NotImplementedError(error_style(self.get_data_length_error_msg()))
        else:
            for i, key in enumerate(self.data_keys):
                self.data_dict[key] = material.data[i]

        self.tolerance: float = 1.0e-6
        self.total_number_of_slips: int = 12
        self.C11 = 169727.0
        self.C12 = 104026.0
        self.C44 = 86000.0
        self.create_elastic_stiffness()
        self.K = 120.0
        self.a = 0.00025
        self.q = 3.0
        self.theta = 0.5
        self.c1 = 2000.0
        self.c2 = 10.0
        self.r0 = 10.0
        self.b = 1.0
        self.Q = 20.0
        self.H = ones(shape=(self.total_number_of_slips, self.total_number_of_slips), dtype=DTYPE)
        self.u_prime = array([1, 0, 0])
        self.v_prime = array([0, 1, 0])
        self.w_prime = array([0, 0, 1])

        self.u = array([0.86602540378, -0.5, 0])
        self.v = array([0.5, 0.86602540378, 0])
        self.w = array([0, 0, 1])

        # self.u = array([1, 0, 0])
        # self.v = array([0, 1, 0])
        # self.w = array([0, 0, 1])

        self.T = get_transformation(self.u, self.v, self.w, self.u_prime, self.v_prime, self.w_prime)
        self.T_vogit = get_voigt_transformation(self.T)

        self.m = array([[0.000000, -0.707107, 0.707107],
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

        self.n = array([[0.577350, 0.577350, 0.577350],
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

        self.m = dot(self.m, self.T)
        self.n = dot(self.n, self.T)
        self.C = dot(dot(self.T_vogit, self.C), transpose(self.T_vogit))
        self.MAX_NITER = 8
        self.create_tangent()

    def create_tangent(self):
        if self.section.type in self.allowed_section_types:
            pass
        else:
            raise NotImplementedError(error_style(self.get_section_type_error_msg()))

    def create_elastic_stiffness(self):
        C11 = self.C11
        C12 = self.C12
        C44 = self.C44
        self.C = array([[C11, C12, C12, 0, 0, 0],
                        [C12, C11, C12, 0, 0, 0],
                        [C12, C12, C11, 0, 0, 0],
                        [0, 0, 0, C44, 0, 0],
                        [0, 0, 0, 0, C44, 0],
                        [0, 0, 0, 0, 0, C44]], dtype=DTYPE)

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
            state_variable['rho_m'] = zeros(shape=self.total_number_of_slips, dtype=DTYPE)
            state_variable['rho_di'] = zeros(shape=self.total_number_of_slips, dtype=DTYPE)

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
            delta_m_e = 0.0

            m_e = deepcopy(state_variable['m_e']) + delta_m_e
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
        state_variable_new['rho_m'] = rho_m
        state_variable_new['rho_di'] = rho_di

        # np.set_printoptions(precision=6, linewidth=256)
        # print(stress)
        # if element_id == 0 and iqp == 0:
        #     print(alpha)
        #     print(T)
        #     print(T_vogit)
        #     print(residual)
        #     print(Omega)
        if self.section.type == 'PlaneStrain':
            ddsdde = delete(delete(ddsdde, [2, 4, 5], axis=0), [2, 4, 5], axis=1)
            stress = delete(stress, [2, 4, 5])

        output = {'stress': stress}

        return ddsdde, output


if __name__ == "__main__":
    # from pyfem.utils.visualization import print_slots_dict
    #
    # print_slots_dict(PlasticCrystal.__slots_dict__)

    from pyfem.Job import Job

    job = Job(r'..\..\..\examples\mechanical\1element\hex20_crystal\Job-1.toml')

    # job.assembly.element_data_list[0].material_data_list[0].show()

    # print(job.assembly.element_data_list[0].qp_state_variables[0]['n_e'])

    job.props.amplitudes[0].show()

    job.run()
