# -*- coding: utf-8 -*-
"""

"""
from sympy import Symbol, exp, latex

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

    rho_m = Symbol('rho_m')
    rho_di = Symbol('rho_di')
    b_s = Symbol('b_s')
    nu_0 = Symbol('nu_0')
    Q_s = Symbol('Q_s')
    k_b = Symbol('k_b')
    T = Symbol('T')
    arrhenius = Q_s / k_b / T
    tau_eff = Symbol('tau_eff')
    tau = Symbol('tau')
    tau_sol = Symbol('tau_sol')
    q = Symbol('q')
    p = Symbol('p')
    lam = Symbol('lambda')
    # d_di = Symbol('d_di')
    nu_clim = Symbol('nu_clim')
    gamma = Symbol('gamma')
    d_min = Symbol('d_min')
    pi = Symbol('pi')
    G = Symbol('G')

    d_di = 3 * G * b_s / 16 / pi / tau
    gamma_dot = rho_m * b_s * nu_0 * exp(-arrhenius * (1 - (tau_eff / tau_sol) ** p) ** q)
    rho_m = gamma_dot / b_s / lam - 2 * rho_m * gamma_dot / b_s * d_di

    print(latex(rho_m))
    print(latex(d_di))
    # print(latex(simplify(rho_m)))
