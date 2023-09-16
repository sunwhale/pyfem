# -*- coding: utf-8 -*-
"""

"""
from copy import deepcopy

import numpy as np
from numpy import zeros, ndarray, sign, dot, array, einsum, eye, ones, maximum, abs, transpose, all, delete, concatenate
from numpy.linalg import solve, inv

from pyfem.fem.Timer import Timer
from pyfem.fem.constants import DTYPE
from pyfem.io.Material import Material
from pyfem.io.Section import Section
from pyfem.materials.BaseMaterial import BaseMaterial
from pyfem.materials.crystal_slip_system import generate_mn
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
        'slip_system_name': ('str', ''),
        'c_over_a': ('str', ''),
        'theta': ('float', '切线系数法参数'),
        'K': ('ndarray', '硬化系数矩阵'),
        'v_0': ('ndarray', '硬化系数矩阵'),
        'p_s': ('ndarray', '硬化系数矩阵'),
        'c_1': ('ndarray', '硬化系数矩阵'),
        'c_2': ('ndarray', '硬化系数矩阵'),
        'r_0': ('ndarray', '硬化系数矩阵'),
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
        'm_s': ('ndarray', '硬化系数矩阵'),
        'n_s': ('ndarray', '硬化系数矩阵'),
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
        self.theta: float = material.data_dict['theta']

        self.elastic: dict = material.data_dict['elastic']
        self.C: ndarray = self.create_elastic_stiffness(self.elastic)

        self.total_number_of_slips: int = 0
        self.slip_system_name: list[str] = material.data_dict['slip_system_name']
        self.c_over_a: list[float] = material.data_dict['c_over_a']

        for i, (name, ca) in enumerate(zip(self.slip_system_name, self.c_over_a)):
            slip_system_number, m_s, n_s = generate_mn('slip', name, ca)
            self.total_number_of_slips += slip_system_number
            K = ones((slip_system_number,), dtype=DTYPE) * material.data_dict['K'][i]
            v_0 = ones((slip_system_number,), dtype=DTYPE) * material.data_dict['v_0'][i]
            p_s = ones((slip_system_number,), dtype=DTYPE) * material.data_dict['p_s'][i]
            c_1 = ones((slip_system_number,), dtype=DTYPE) * material.data_dict['c_1'][i]
            c_2 = ones((slip_system_number,), dtype=DTYPE) * material.data_dict['c_2'][i]
            r_0 = ones((slip_system_number,), dtype=DTYPE) * material.data_dict['r_0'][i]
            b = ones((slip_system_number,), dtype=DTYPE) * material.data_dict['b'][i]
            Q = ones((slip_system_number,), dtype=DTYPE) * material.data_dict['Q'][i]
            if i == 0:
                self.m_s: ndarray = m_s
                self.n_s: ndarray = n_s
                self.K: ndarray = K
                self.v_0: ndarray = v_0
                self.p_s: ndarray = p_s
                self.c_1: ndarray = c_1
                self.c_2: ndarray = c_2
                self.r_0: ndarray = r_0
                self.b: ndarray = b
                self.Q: ndarray = Q
            else:
                self.m_s = concatenate((self.m_s, m_s))
                self.n_s = concatenate((self.n_s, n_s))
                self.K = concatenate((self.K, K))
                self.v_0 = concatenate((self.v_0, v_0))
                self.p_s = concatenate((self.p_s, p_s))
                self.c_1 = concatenate((self.c_1, c_1))
                self.c_2 = concatenate((self.c_2, c_2))
                self.r_0 = concatenate((self.r_0, r_0))
                self.b = concatenate((self.b, b))
                self.Q = concatenate((self.Q, Q))

        self.H = ones(shape=(self.total_number_of_slips, self.total_number_of_slips), dtype=DTYPE)

        self.u_global: ndarray = array(section.data_dict['u_global'])
        self.v_global: ndarray = array(section.data_dict['v_global'])
        self.w_global: ndarray = array(section.data_dict['w_global'])

        self.u_grain: ndarray = array(section.data_dict['u_grain'])
        self.v_grain: ndarray = array(section.data_dict['v_grain'])
        self.w_grain: ndarray = array(section.data_dict['w_grain'])

        self.T: ndarray = get_transformation(self.u_grain, self.v_grain, self.w_grain, self.u_global, self.v_global, self.w_global)
        self.T_vogit: ndarray = get_voigt_transformation(self.T)

        self.m_s = dot(self.m_s, self.T)
        self.n_s = dot(self.n_s, self.T)
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
        v_0 = self.v_0
        p_s = self.p_s
        dt = timer.dtime
        theta = self.theta
        c_1 = self.c_1
        c_2 = self.c_2
        r_0 = self.r_0
        b = self.b
        Q = self.Q
        H = self.H
        C = self.C
        m_s = self.m_s
        n_s = self.n_s

        if state_variable == {} or timer.time0 == 0.0:
            state_variable['m_s'] = m_s
            state_variable['n_s'] = n_s
            m_sxn_s = transpose(array([m_s[:, 0] * n_s[:, 0],
                                       m_s[:, 1] * n_s[:, 1],
                                       m_s[:, 2] * n_s[:, 2],
                                       2.0 * m_s[:, 0] * n_s[:, 1],
                                       2.0 * m_s[:, 0] * n_s[:, 2],
                                       2.0 * m_s[:, 1] * n_s[:, 2]]))
            n_sxm_s = transpose(array([n_s[:, 0] * m_s[:, 0],
                                       n_s[:, 1] * m_s[:, 1],
                                       n_s[:, 2] * m_s[:, 2],
                                       2.0 * n_s[:, 0] * m_s[:, 1],
                                       2.0 * n_s[:, 0] * m_s[:, 2],
                                       2.0 * n_s[:, 1] * m_s[:, 2]]))
            P = 0.5 * (m_sxn_s + n_sxm_s)
            state_variable['stress'] = zeros(shape=6, dtype=DTYPE)
            state_variable['tau'] = dot(P, state_variable['stress'])
            state_variable['gamma'] = zeros(shape=self.total_number_of_slips, dtype=DTYPE)
            state_variable['rho'] = zeros(shape=self.total_number_of_slips, dtype=DTYPE)
            state_variable['alpha'] = zeros(shape=self.total_number_of_slips, dtype=DTYPE)
            state_variable['r'] = zeros(shape=self.total_number_of_slips, dtype=DTYPE) + r_0

        rho = deepcopy(state_variable['rho'])
        m_s = deepcopy(state_variable['m_s'])
        n_s = deepcopy(state_variable['n_s'])
        gamma = deepcopy(state_variable['gamma'])
        stress = deepcopy(state_variable['stress'])
        tau = deepcopy(state_variable['tau'])
        alpha = deepcopy(state_variable['alpha'])
        r = deepcopy(state_variable['r'])

        delta_gamma = zeros(shape=self.total_number_of_slips, dtype=DTYPE)

        is_convergence = False

        for niter in range(self.MAX_NITER):
            m_sxn_s = transpose(array([m_s[:, 0] * n_s[:, 0],
                                       m_s[:, 1] * n_s[:, 1],
                                       m_s[:, 2] * n_s[:, 2],
                                       2.0 * m_s[:, 0] * n_s[:, 1],
                                       2.0 * m_s[:, 0] * n_s[:, 2],
                                       2.0 * m_s[:, 1] * n_s[:, 2]]))

            n_sxm_s = transpose(array([n_s[:, 0] * m_s[:, 0],
                                       n_s[:, 1] * m_s[:, 1],
                                       n_s[:, 2] * m_s[:, 2],
                                       2.0 * n_s[:, 0] * m_s[:, 1],
                                       2.0 * n_s[:, 0] * m_s[:, 2],
                                       2.0 * n_s[:, 1] * m_s[:, 2]]))

            P = 0.5 * (m_sxn_s + n_sxm_s)
            Omega = 0.5 * (m_sxn_s - n_sxm_s)
            Omega[:, 3:] *= 0.5

            # S = dot(P, C) + Omega * stress - stress * Omega
            S = dot(P, C)

            X = (abs(tau - alpha) - r) / K
            gamma_dot = v_0 * maximum(X, 0.0) ** p_s * sign(tau - alpha)

            if niter == 0:
                gamma_dot_t = deepcopy(gamma_dot)

            term1 = dt * theta
            term2 = term1 * v_0 * p_s * maximum(X, 0.0) ** (p_s - 1.0) / K
            term3 = term1 * maximum(X, 0) * v_0 * p_s * maximum(X, 0.0) ** (p_s - 1.0) / K
            term4 = einsum('ik, jk->ij', S, P)

            A = eye(self.total_number_of_slips, dtype=DTYPE)
            A += term2 * term4
            # A += H * term3 * sign(gamma_dot) * sign(tau - alpha)
            A += term2 * (c_1 - c_2 * alpha * sign(gamma_dot)) * eye(self.total_number_of_slips, dtype=DTYPE)
            A += term2 * b * Q * H * (1.0 - b * rho) * sign(tau - alpha) * sign(gamma_dot)

            if niter == 0:
                rhs = dt * gamma_dot + term2 * dot(S, dstrain)
            else:
                rhs = term1 * (gamma_dot - gamma_dot_t) + gamma_dot_t * dt - delta_gamma

            d_delta_gamma = solve(transpose(A), rhs)
            delta_gamma += d_delta_gamma

            delta_elastic_strain = dstrain - dot(delta_gamma, P)
            delta_tau = dot(S, delta_elastic_strain)
            delta_stress = dot(C, delta_elastic_strain)
            delta_alpha = c_1 * delta_gamma - c_2 * abs(delta_gamma) * alpha
            delta_rho = (1.0 - b * rho) * abs(delta_gamma)
            delta_r = b * Q * dot(H, delta_rho)

            gamma = deepcopy(state_variable['gamma']) + delta_gamma
            tau = deepcopy(state_variable['tau']) + delta_tau
            stress = deepcopy(state_variable['stress']) + delta_stress
            alpha = deepcopy(state_variable['alpha']) + delta_alpha
            rho = deepcopy(state_variable['rho']) + delta_rho
            r = deepcopy(state_variable['r']) + delta_r

            X = (abs(tau - alpha) - r) / K
            gamma_dot = v_0 * maximum(X, 0.0) ** p_s * sign(tau - alpha)
            residual = dt * theta * gamma_dot + dt * (1.0 - theta) * gamma_dot_t - delta_gamma

            # if element_id == 0 and iqp == 0:
            #     print('residual', residual)

            if all(residual < self.tolerance):
                is_convergence = True
                break

        ddgdde = term2.reshape((self.total_number_of_slips, 1)) * S
        ddgdde = dot(inv(A), ddgdde)
        ddsdde = C - einsum('ki, kj->ij', S, ddgdde)

        if not is_convergence:
            timer.is_reduce_dtime = True

        state_variable_new['m_s'] = m_s
        state_variable_new['n_s'] = n_s
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
