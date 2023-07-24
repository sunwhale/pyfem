# -*- coding: utf-8 -*-
"""

"""
from sympy import Symbol, exp, latex, simplify, diff
import math

if __name__ == '__main__':
    # x = symbols('x')
    # expr = x**3 + 2*x**2 + x + 1
    #
    # # 对表达式求导
    # derivative = diff(expr, x)
    #
    # # 打印结果
    # print(latex(derivative))

    # m_1, m_2, m_3 = symbols('m_1, m_2, m_3')
    # n_1, n_2, n_3 = symbols('n_1, n_2, n_3')
    #
    # Fe_11, Fe_12, Fe_13, Fe_21, Fe_22, Fe_23, Fe_31, Fe_32, Fe_33 = symbols(
    #     'F^e_11 F^e_12 F^e_13 F^e_21 F^e_22 F^e_23 F^e_31 F^e_32 F^e_33')
    # Fe = Matrix([[Fe_11, Fe_12, Fe_13], [Fe_21, Fe_22, Fe_23], [Fe_31, Fe_32, Fe_33]])
    #
    # Fp_11, Fp_12, Fp_13, Fp_21, Fp_22, Fp_23, Fp_31, Fp_32, Fp_33 = symbols(
    #     'F^p_11 F^p_12 F^p_13 F^p_21 F^p_22 F^p_23 F^p_31 F^p_32 F^p_33')
    # Fp = Matrix([[Fp_11, Fp_12, Fp_13], [Fp_21, Fp_22, Fp_23], [Fp_31, Fp_32, Fp_33]])
    #
    # F = Fe * Fp
    #
    # # m = Matrix([m_1, m_2, m_3])
    # # n = Matrix([n_1, n_2, n_3])
    #
    # me_1, me_2, me_3 = symbols('m^e_1, m^e_2, m^e_3')
    # ne_1, ne_2, ne_3 = symbols('n^e_1, n^e_2, n^e_3')
    #
    # me = Matrix([me_1, me_2, me_3])
    # ne = Matrix([ne_1, ne_2, ne_3])
    #
    # me_dyadic_ne = tensorcontraction(tensorproduct(me, ne), me.shape)
    # ne_dyadic_me = tensorcontraction(tensorproduct(ne, me), ne.shape)
    #
    # P = 1 / 2 * (me_dyadic_ne + ne_dyadic_me)

    # print(P)

    # Omega = 1 / 2 * (me_dyadic_ne - ne_dyadic_me)

    # tau, g = symbols('tau g')
    #
    # X = tau / g
    #
    # f = Piecewise((X * X ** (n - 1), X >= 0), (X * -X ** (n - 1), X < 0))
    # df = sign(X) * n * Abs(X) ** (n - 1)

    # derivative = simplify(diff(f, tau))
    # print(latex(derivative))

    # derivative = simplify(diff(f, g))
    # print(latex(derivative))

    # Constants
    b_s = Symbol('b_s')
    nu_0 = Symbol('nu_0')
    Q_s = Symbol('Q_s')
    k_b = Symbol('k_b')
    T = Symbol('T')
    pi = Symbol('pi')
    G = Symbol('G')
    h = Symbol('h_alpha_beta')

    # Parameters
    q = Symbol('q')
    p = Symbol('p')
    C_clim = Symbol('C_clim')
    D_0 = Symbol('D_0')
    Q_clim = Symbol('Q_clim')
    OMEGA_clim = Symbol('OMEGA_clim')
    i_slip = Symbol('i_slip')
    # Variables
    rho_m = Symbol('rho_m')
    rho_di = Symbol('rho_di')
    # tau_eff = Symbol('tau_eff')
    tau = Symbol('tau')
    tau_sol = Symbol('tau_sol')
    gamma = Symbol('gamma')
    sumRho = Symbol('sumRho')
    lam = Symbol('lambda')
    # nu_clim = Symbol('nu_clim')

    A = G * b_s
    B = 3 * G * D_0 * OMEGA_clim / 2 / pi / k_b / T * exp(-Q_clim / k_b / T)
    arrhenius = Q_s / k_b / T
    d_di = 3 * G * b_s / 16 / pi / tau
    d_min = b_s * C_clim
    nu_clim = B / (d_di + d_min)
    tau_pass = A * sumRho ** (1/2)
    tau_eff = tau - tau_pass

    #  for i in Nslip:
    #      for j in Nslip:
    #        sumRho =

    gamma_dot = rho_m * b_s * nu_0 * exp(-arrhenius * (1 - (tau_eff / tau_sol) ** p) ** q)
    rho_m_dot = gamma_dot / b_s / lam - 2 * rho_m * gamma_dot / b_s * d_di
    rho_di_dot = 2 * gamma_dot * (d_di - d_min) - 2 * gamma_dot * rho_di * d_min / b_s - 4 * rho_di * nu_clim / (
                d_di - d_min)

    # beta = 6
    # rho_m = [Symbol('\\rho_{m}^{(' + str(i) + ')}') for i in range(1, beta+1)]
    # rho_di = [Symbol('\\rho_{di}^{(' + str(i) + ')}') for i in range(1, beta+1)]
    # h_matrix = [[Symbol('h_{' + str(i) + ',' + str(j) + '}') for j in range(1, beta+1)] for i in range(1, beta+1)]
    #
    # sum_rho_m = rho_m[0]
    # for i in range(1, beta):
    #     sum_rho_m += rho_m[i]
    #
    # expression = h_matrix[0][0]*(rho_m[0]+rho_di[0])
    # for i in range(1, beta):
    #     expression += h_matrix[0][i]*(rho_m[i]+rho_di[i])
    #
    # expression = expression ** 0.5

    # print(latex(gamma_dot))
    # print(latex(rho_m_dot))
    # print(latex(rho_di_dot))
    # print(latex(simplify(gamma_dot)))
    # print(latex(simplify(rho_m_dot)))
    # print(latex(simplify(rho_di_dot)))

