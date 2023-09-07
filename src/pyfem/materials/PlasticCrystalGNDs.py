# -*- coding: utf-8 -*-
"""

"""
from copy import deepcopy

import numpy as np
from numpy import pi, zeros, exp, ndarray, sqrt, sign, dot, array, einsum, eye, ones, maximum, abs, transpose, all, sum
from numpy.linalg import solve, inv

from pyfem.fem.Timer import Timer
from pyfem.fem.constants import DTYPE
from pyfem.io.Material import Material
from pyfem.io.Section import Section
from pyfem.materials.BaseMaterial import BaseMaterial
from pyfem.materials.ElasticIsotropic import get_stiffness_from_young_poisson
from pyfem.utils.colors import error_style
from pyfem.utils.mechanics import get_transformation, get_voigt_transformation


class PlasticCrystalGNDs(BaseMaterial):
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
        'C11': ('float', '弹性矩阵系数'),
        'C12': ('float', '弹性矩阵系数'),
        'C44': ('float', '弹性矩阵系数'),
        'C': ('ndarray', '旋转矩阵'),
        'theta': ('float', '切线系数法参数'),
        'H': ('ndarray', '硬化系数矩阵'),
        'tau_sol': ('float', '固溶强度'),
        'v_0': ('float', '位错滑移速度'),
        'b_s': ('float', '位错滑移柏氏矢量长度'),
        'Q_s': ('float', '位错滑移激活能'),
        'p_s': ('float', '位错滑移阻力拟合参数'),
        'q_s': ('float', '位错滑移阻力拟合参数'),
        'k_b': ('float', '玻尔兹曼常数'),
        'd_grain': ('float', '平均晶粒尺寸'),
        'i_slip': ('float', '平均位错间隔拟合参数'),
        'c_anni': ('float', '位错消除拟合参数'),
        'Q_climb': ('float', '位错攀移激活能'),
        'D_0': ('float', '自扩散系数因子'),
        'Omega_climb': ('float', '位错攀移激活体积'),
        'G': ('float', '剪切模量'),
        'temperature': ('float', '温度'),
        'u': ('ndarray', 'u方向矢量'),
        'v': ('ndarray', 'v方向矢量'),
        'w': ('ndarray', 'w方向矢量'),
        'u_prime': ('ndarray', '特征u方向矢量'),
        'v_prime': ('ndarray', '特征v方向矢量'),
        'w_prime': ('ndarray', '特征w方向矢量'),
        'T': ('ndarray', '旋转矩阵'),
        'T_vogit': ('ndarray', 'Vogit形式旋转矩阵'),
        'm': ('ndarray', '特征滑移系滑移方向'),
        'n': ('ndarray', '特征滑移系滑移面法向'),
        'MAX_NITER': ('ndarray', '最大迭代次数'),
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
        self.theta: float = 0.5
        self.total_number_of_slips: int = 12
        self.C11 = 175000.0
        self.C12 = 115000.0
        self.C44 = 135000.0
        self.create_elastic_stiffness()

        self.tau_sol = 130.0
        self.v_0 = 0.1 * 1e3
        self.b_s = 2.56e-7
        self.Q_s = 3.5e-19 * 1e3

        self.p_s = 1.15
        self.q_s = 1.0
        self.k_b = 1.38e-23 * 1e3
        self.d_grain = 5.0e-3
        self.i_slip = 30.0
        self.c_anni = 2.0
        self.Q_climb = 3.0e-19 * 1e3
        self.D_0 = 40.0
        self.Omega_climb = 1.5 * self.b_s ** 3
        self.G = 79000.0
        self.temperature = 298.13

        self.H = ones(shape=(self.total_number_of_slips, self.total_number_of_slips), dtype=DTYPE)
        self.u_prime = array([1, 0, 0])
        self.v_prime = array([0, 1, 0])
        self.w_prime = array([0, 0, 1])

        self.u = array([0.86602540378, -0.5, 0])
        self.v = array([0.5, 0.86602540378, 0])
        self.w = array([0, 0, 1])

        self.u = array([1, 0, 0])
        self.v = array([0, 1, 0])
        self.w = array([0, 0, 1])

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

        np.set_printoptions(precision=12, linewidth=256, suppress=True)

        C = self.C
        total_number_of_slips = self.total_number_of_slips

        tau_sol = self.tau_sol
        v_0 = self.v_0
        b_s = self.b_s
        Q_s = self.Q_s

        p_s = self.p_s
        q_s = self.q_s
        k_b = self.k_b
        d_grain = self.d_grain
        i_slip = self.i_slip
        c_anni = self.c_anni
        Q_climb = self.Q_climb
        D_0 = self.D_0
        Omega_climb = self.Omega_climb
        G = self.G
        temperature = self.temperature

        d_min = c_anni * b_s

        dt = timer.dtime
        theta = self.theta

        H = self.H
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
            state_variable['gamma'] = zeros(shape=total_number_of_slips, dtype=DTYPE)
            state_variable['tau_pass'] = zeros(shape=total_number_of_slips, dtype=DTYPE)
            state_variable['rho_m'] = zeros(shape=total_number_of_slips, dtype=DTYPE) + 1e-6
            state_variable['rho_di'] = zeros(shape=total_number_of_slips, dtype=DTYPE)

        rho_m = deepcopy(state_variable['rho_m'])
        rho_di = deepcopy(state_variable['rho_di'])
        m_e = deepcopy(state_variable['m_e'])
        n_e = deepcopy(state_variable['n_e'])
        gamma = deepcopy(state_variable['gamma'])
        stress = deepcopy(state_variable['stress'])
        tau = deepcopy(state_variable['tau'])

        delta_gamma = zeros(shape=total_number_of_slips, dtype=DTYPE)
        delta_stress = zeros(shape=6, dtype=DTYPE)
        delta_tau = zeros(shape=total_number_of_slips, dtype=DTYPE)
        delta_tau_pass = zeros(shape=total_number_of_slips, dtype=DTYPE)
        delta_rho_m = zeros(shape=total_number_of_slips, dtype=DTYPE)
        delta_rho_di = zeros(shape=total_number_of_slips, dtype=DTYPE)

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

            rho = rho_di + rho_m
            tau_pass = G * b_s * sqrt(dot(H, rho))

            X = (abs(tau) - tau_pass) / tau_sol
            X_bracket = maximum(X, 0.0)
            X_heaviside = sign(X_bracket)
            A_s = Q_s / k_b / temperature

            d_di = 3.0 * G * b_s / 16.0 / pi * abs(tau)
            one_over_lambda = 1.0 / d_grain + 1.0 / i_slip * tau_pass / G / b_s
            v_climb = 3.0 * G * D_0 * Omega_climb / (2.0 * pi * k_b * temperature * (d_di + d_min)) * exp(-Q_climb / k_b / temperature)
            gamma_dot = rho_m * b_s * v_0 * exp(-A_s * (1.0 - X_bracket ** p_s) ** q_s) * sign(tau)

            if niter == 0:
                gamma_dot_init = deepcopy(gamma_dot)

            term1 = dt * theta
            term2 = A_s * p_s * q_s * gamma_dot * X_bracket ** (p_s - 1.0) * (1.0 - X_bracket) ** (q_s - 1.0) * sign(tau)
            term3 = X_heaviside / tau_sol
            term4 = einsum('ik, jk->ij', S, P)
            term5 = one_over_lambda / b_s - 2.0 * d_min * rho_m / b_s
            term6 = one_over_lambda / b_s - 2.0 * d_min * rho / b_s
            term7 = 4.0 * rho_di * v_climb / (d_di - d_min)
            term8 = (G * b_s) ** 2 / (2.0 * tau_pass)

            I = eye(total_number_of_slips, dtype=DTYPE)
            A = deepcopy(I)
            A -= term1 * term2 * term5 * b_s * v_0 * exp(-A_s * (1.0 - X_bracket ** p_s) ** q_s) * I
            A += term1 * term2 * term3 * term4 * sign(tau)
            A += term1 * term2 * term3 * term8 * dot(H, term6 * sign(tau) * I)

            if niter == 0:
                rhs = dt * gamma_dot + term1 * term2 * term3 * sign(tau) * dot(S, dstrain) + term1 * term2 * term3 * term8 * dot(H, term7)
                # rhs = dt * gamma_dot + term1 * term2 * term3 * sign(tau) * dot(S, dstrain)
            else:
                rhs = dt * theta * (gamma_dot - gamma_dot_init) + gamma_dot_init * dt - delta_gamma

            d_delta_gamma = solve(transpose(A), rhs)
            delta_gamma += d_delta_gamma

            delta_elastic_strain = dstrain - dot(delta_gamma, P)
            delta_tau = dot(S, delta_elastic_strain)
            delta_stress = dot(C, delta_elastic_strain)
            delta_rho_m = (one_over_lambda / b_s - 2.0 * d_di * rho_m / b_s) * abs(delta_gamma)
            delta_rho_di = 2.0 * (rho_m * (d_di - d_min) - rho_di * d_min) / b_s * abs(delta_gamma) - term7

            delta_m_e = 0.0
            delta_n_e = 0.0

            m_e = deepcopy(state_variable['m_e']) + delta_m_e
            n_e = deepcopy(state_variable['n_e']) + delta_n_e
            gamma = deepcopy(state_variable['gamma']) + delta_gamma
            tau = deepcopy(state_variable['tau']) + delta_tau
            stress = deepcopy(state_variable['stress']) + delta_stress
            rho_m = deepcopy(state_variable['rho_m']) + delta_rho_m
            rho_di = deepcopy(state_variable['rho_di']) + delta_rho_di

            X = (abs(tau) - tau_pass) / tau_sol
            X_bracket = maximum(X, 0.0)
            gamma_dot = rho_m * b_s * v_0 * exp(-A_s * (1.0 - X_bracket ** p_s) ** q_s) * sign(tau)
            residual = dt * theta * gamma_dot + dt * (1.0 - theta) * gamma_dot_init - delta_gamma

            if element_id == 0 and iqp == 0:
                print('residual', residual)

            if all(residual < self.tolerance):
                is_convergence = True
                break

        ddgdde = (term1 * term2 * term3 * sign(tau)).reshape((total_number_of_slips, 1)) * S
        ddgdde = dot(inv(A), ddgdde)
        ddsdde = C - einsum('ki, kj->ij', S, ddgdde)
        # ddsdde = C

        # if element_id == 0 and iqp == 0:
            # print('A', A)
            # print('tau_sol', tau_sol)
            # print('tau_pass', tau_pass)
            # print('rho', rho)
            # print('stress', stress)
            # print('tau', tau)
            # print('gamma', gamma)
            # print('rho_m', rho_m)
            # print('rho_di', rho_di)

        state_variable_new['m_e'] = m_e
        state_variable_new['n_e'] = n_e
        state_variable_new['stress'] = stress
        state_variable_new['gamma'] = gamma
        state_variable_new['tau'] = tau
        state_variable_new['rho_m'] = rho_m
        state_variable_new['rho_di'] = rho_di

        output = {'stress': stress}

        np.set_printoptions(precision=6, linewidth=256)

        return ddsdde, output


if __name__ == "__main__":
    # from pyfem.utils.visualization import print_slots_dict
    #
    # print_slots_dict(PlasticCrystal.__slots_dict__)

    from pyfem.Job import Job

    job = Job(r'..\..\..\examples\mechanical\1element\hex20_crystal_GNDs\Job-1.toml')

    # job.assembly.element_data_list[0].material_data_list[0].show()

    # print(job.assembly.element_data_list[0].qp_state_variables[0]['n_e'])

    job.props.amplitudes[0].show()

    job.run()
