# -*- coding: utf-8 -*-
"""

"""
from copy import deepcopy

import numpy as np
from numpy import pi, zeros, exp, ndarray, sqrt, sign, dot, array, einsum, eye, ones, maximum, abs, transpose, all, delete, insert
from numpy.linalg import solve, inv

from pyfem.fem.Timer import Timer
from pyfem.fem.constants import DTYPE
from pyfem.io.Material import Material
from pyfem.io.Section import Section
from pyfem.materials.BaseMaterial import BaseMaterial
from pyfem.utils.colors import error_style
from pyfem.utils.mechanics import get_transformation, get_voigt_transformation


class PlasticCrystalGNDs(BaseMaterial):
    r"""
    晶体塑性材料。

    支持的截面属性：('Volume', 'PlaneStress', 'PlaneStrain')

    :ivar tolerance: 判断屈服的误差容限
    :vartype tolerance: float

    :ivar total_number_of_slips: 总的滑移系数量
    :vartype total_number_of_slips: int

    :ivar C11: 弹性矩阵系数
    :vartype C11: float

    :ivar C12: 弹性矩阵系数
    :vartype C12: float

    :ivar C44: 弹性矩阵系数
    :vartype C44: float

    :ivar C: 旋转矩阵
    :vartype C: ndarray

    :ivar theta: 切线系数法参数
    :vartype theta: float

    :ivar H: 硬化系数矩阵
    :vartype H: ndarray

    :ivar tau_sol: 固溶强度
    :vartype tau_sol: float

    :ivar v_0: 位错滑移速度
    :vartype v_0: float

    :ivar b_s: 位错滑移柏氏矢量长度
    :vartype b_s: float

    :ivar Q_s: 位错滑移激活能
    :vartype Q_s: float

    :ivar p_s: 位错滑移阻力拟合参数
    :vartype p_s: float

    :ivar q_s: 位错滑移阻力拟合参数
    :vartype q_s: float

    :ivar k_b: 玻尔兹曼常数
    :vartype k_b: float

    :ivar d_grain: 平均晶粒尺寸
    :vartype d_grain: float

    :ivar i_slip: 平均位错间隔拟合参数
    :vartype i_slip: float

    :ivar c_anni: 位错消除拟合参数
    :vartype c_anni: float

    :ivar Q_climb: 位错攀移激活能
    :vartype Q_climb: float

    :ivar D_0: 自扩散系数因子
    :vartype D_0: float

    :ivar Omega_climb: 位错攀移激活体积
    :vartype Omega_climb: float

    :ivar G: 剪切模量
    :vartype G: float

    :ivar temperature: 温度
    :vartype temperature: float

    :ivar u: u方向矢量
    :vartype u: ndarray

    :ivar v: v方向矢量
    :vartype v: ndarray

    :ivar w: w方向矢量
    :vartype w: ndarray

    :ivar u_prime: 特征u方向矢量
    :vartype u_prime: ndarray

    :ivar v_prime: 特征v方向矢量
    :vartype v_prime: ndarray

    :ivar w_prime: 特征w方向矢量
    :vartype w_prime: ndarray

    :ivar T: 旋转矩阵
    :vartype T: ndarray

    :ivar T_vogit: Vogit形式旋转矩阵
    :vartype T_vogit: ndarray

    :ivar m: 特征滑移系滑移方向
    :vartype m: ndarray

    :ivar n: 特征滑移系滑移面法向
    :vartype n: ndarray

    :ivar MAX_NITER: 最大迭代次数
    :vartype MAX_NITER: ndarray

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
        'm_s': ('ndarray', '特征滑移系滑移方向'),
        'n_s': ('ndarray', '特征滑移系滑移面法向'),
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
        self.MAX_NITER = 8
        self.theta: float = 0.5
        self.total_number_of_slips: int = 12

        self.C11 = 107.0e9
        self.C12 = 52.0e9
        self.C44 = 26.0e9
        self.create_elastic_stiffness()

        self.tau_sol = 52.0e6
        self.v_0 = 1.0e-4
        self.b_s = 2.546e-10
        self.Q_s = 8.36e-20

        self.p_s = 0.8
        self.q_s = 1.6
        self.k_b = 1.38e-23
        self.d_grain = 15.25e-6
        self.i_slip = 28.0
        self.c_anni = 7.0
        self.Q_climb = 1.876e-19
        self.D_0 = 6.23e-4
        self.Omega_climb = 4.0 * self.b_s ** 3
        self.G = 26.0e9
        self.temperature = 298.13

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

        self.m_s = array([[0.000000, -0.707107, 0.707107],
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

        self.n_s = array([[0.577350, 0.577350, 0.577350],
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

        self.m_s = dot(self.m_s, self.T)
        self.n_s = dot(self.n_s, self.T)
        self.C = dot(dot(self.T_vogit, self.C), transpose(self.T_vogit))
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
            state_variable['gamma'] = zeros(shape=total_number_of_slips, dtype=DTYPE)
            state_variable['tau_pass'] = zeros(shape=total_number_of_slips, dtype=DTYPE)
            state_variable['rho_m'] = zeros(shape=total_number_of_slips, dtype=DTYPE) + 1e12
            state_variable['rho_di'] = zeros(shape=total_number_of_slips, dtype=DTYPE) + 1.0

        rho_m = deepcopy(state_variable['rho_m'])
        rho_di = deepcopy(state_variable['rho_di'])
        m_s = deepcopy(state_variable['m_s'])
        n_s = deepcopy(state_variable['n_s'])
        gamma = deepcopy(state_variable['gamma'])
        stress = deepcopy(state_variable['stress'])
        tau = deepcopy(state_variable['tau'])

        delta_gamma = zeros(shape=total_number_of_slips, dtype=DTYPE)

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

            rho = rho_di + rho_m
            tau_pass = G * b_s * sqrt(dot(H, rho))

            X = (abs(tau) - tau_pass) / tau_sol
            X_bracket = maximum(X, 0.0) + self.tolerance
            X_heaviside = sign(X_bracket)
            A_s = Q_s / k_b / temperature

            d_di = 3.0 * G * b_s / (16.0 * pi * (abs(tau) + self.tolerance))
            one_over_lambda = 1.0 / d_grain + 1.0 / i_slip * tau_pass / G / b_s
            v_climb = 3.0 * G * D_0 * Omega_climb / (2.0 * pi * k_b * temperature * (d_di + d_min)) \
                      * exp(-Q_climb / k_b / temperature)
            gamma_dot = rho_m * b_s * v_0 * exp(-A_s * (1.0 - X_bracket ** p_s) ** q_s) * sign(tau)

            if niter == 0:
                gamma_dot_init = deepcopy(gamma_dot)

            term1 = dt * theta
            term2 = A_s * p_s * q_s * gamma_dot * X_bracket ** (p_s - 1.0) * (1.0 - X_bracket) ** (q_s - 1.0) \
                    * sign(tau)
            term3 = X_heaviside / tau_sol
            term4 = einsum('ik, jk->ij', S, P)
            term5 = one_over_lambda / b_s - 2.0 * d_min * rho_m / b_s
            term6 = one_over_lambda / b_s - 2.0 * d_min * rho / b_s
            term7 = 4.0 * rho_di * v_climb / (d_di - d_min) * dt
            term8 = (G * b_s) ** 2 / (2.0 * tau_pass)

            I = eye(total_number_of_slips, dtype=DTYPE)
            A = deepcopy(I)
            A -= term1 * term2 * term5 * b_s * v_0 * exp(-A_s * (1.0 - X_bracket ** p_s) ** q_s) * I
            A += term1 * term2 * term3 * term4 * sign(tau)
            A += term1 * term2 * term3 * term8 * dot(H, term6 * sign(tau) * I)

            if niter == 0:
                rhs = dt * gamma_dot + term1 * term2 * term3 * sign(tau) * dot(S, dstrain) \
                      + term1 * term2 * term3 * term8 * dot(H, term7)
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
            delta_m_s = 0.0
            delta_n_s = 0.0

            m_s = deepcopy(state_variable['m_s']) + delta_m_s
            n_s = deepcopy(state_variable['n_s']) + delta_n_s
            gamma = deepcopy(state_variable['gamma']) + delta_gamma
            tau = deepcopy(state_variable['tau']) + delta_tau
            stress = deepcopy(state_variable['stress']) + delta_stress
            rho_m = deepcopy(state_variable['rho_m']) + delta_rho_m
            rho_di = deepcopy(state_variable['rho_di']) + delta_rho_di

            X = (abs(tau) - tau_pass) / tau_sol
            X_bracket = maximum(X, 0.0)
            gamma_dot = rho_m * b_s * v_0 * exp(-A_s * (1.0 - X_bracket ** p_s) ** q_s) * sign(tau)
            residual = dt * theta * gamma_dot + dt * (1.0 - theta) * gamma_dot_init - delta_gamma

            # if element_id == 0 and iqp == 0:
            #     print('residual', residual)

            if all(residual < self.tolerance):
                is_convergence = True
                break

        ddgdde = (term1 * term2 * term3 * sign(tau)).reshape((total_number_of_slips, 1)) * S
        ddgdde = dot(inv(A), ddgdde)
        ddsdde = C - einsum('ki, kj->ij', S, ddgdde)

        # if element_id == 0 and iqp == 0:
        #     print('rho_m', rho_m)
        #     print('rho_di', rho_di)

        if not is_convergence:
            timer.is_reduce_dtime = True

        state_variable_new['m_s'] = m_s
        state_variable_new['n_s'] = n_s
        state_variable_new['stress'] = stress
        state_variable_new['gamma'] = gamma
        state_variable_new['tau'] = tau
        state_variable_new['rho_m'] = rho_m
        state_variable_new['rho_di'] = rho_di

        if self.section.type == 'PlaneStrain':
            ddsdde = delete(delete(ddsdde, [2, 4, 5], axis=0), [2, 4, 5], axis=1)
            stress = delete(stress, [2, 4, 5])

        output = {'stress': stress}

        return ddsdde, output


if __name__ == "__main__":
    # from pyfem.utils.visualization import print_slots_dict
    #
    # print_slots_dict(PlasticCrystalGNDs.__slots_dict__)

    from pyfem.Job import Job

    # job = Job(r'..\..\..\examples\mechanical\1element\hex20_crystal_GNDs\Job-1.toml')
    job = Job(r'..\..\..\examples\mechanical\plane_crystal_GNDs\Job-1.toml')

    job.run()
