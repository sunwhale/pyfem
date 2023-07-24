# -*- coding: utf-8 -*-
"""

"""
from sympy import Symbol, exp, latex, simplify
from numpy import sqrt

if __name__ == '__main__':
    # x = symbols('x')
    # expr = x**3 + 2*x**2 + x + 1
    #
    # # 对表达式求导
    # derivative = diff(expr, x)
    #
    # # 打印结果
    # print(latex(derivative))

    rho_m = Symbol('rho_m')   # the mobile dislocation density
    rho_di = Symbol('rho_di')  # dipole dislocation density
    b_s = Symbol('b_s')  # the Burgers vector length for slip
    nu_0 = Symbol('nu_0')  # the reference velocity
    Q_s = Symbol('Q_s')  # the activation energy for dislocation slip
    k_b = Symbol('k_b')  # the Boltzmann constant
    T = Symbol('T')  # temperature
    arrhenius = Q_s / k_b / T  # 阿伦尼乌斯方程方程
    tau_eff = Symbol('tau_eff')  # the effective resolved shear stress
    # tau_pass = Symbol('tau_pass')  # the athermal slip resistance
    tau = Symbol('tau')   # shear stress
    tau_sol = Symbol('tau_sol')  # the solid solution strength
    q = Symbol('q')  # the fitting parameters
    p = Symbol('p')  # the fitting parameters
    lam = Symbol('lambda')  # the mean free path of dislocation slip
    nu_clim = Symbol('nu_clim')  # the dislocation climbing velocity
    Omega_clim = Symbol('Omega_clim')  # 位错攀移的激活体积
    Q_clim = Symbol('Q_clim')  # 位错攀移的激活能
    D_0 = Symbol('D_0')   # self-diffusion coefficient of materials
    c_clim = Symbol('c_clim')   # the fitting parameter discriping the dislocation annihilation
    d = Symbol('d')   # the average grain size
    G = Symbol('G')   # the average grain size
    gamma = Symbol('gamma')
    i_slip = Symbol('i_slip')   # the average dislication slip spacing
    # d_min = Symbol('d_min')
    # d_di = Symbol('d_di')  # dipole dislocation density
    h_alpha_beta = Symbol('h_alpha_beta')
    pi = Symbol('pi')
    arrhenius_clim = Q_clim / k_b / T  # 阿伦尼乌斯方程方程
    # rho_sum = Symbol('rho_sum')  # the total dislocation density



    # rho_sum = sum(h_alpha_beta * (rho_m + rho_di))  # the total dislocation density
    #
    # tau_pass = G * b_s * sqrt(rho_sum)  # the athermal slip resistance
    # d_di = 3 * G * b_s / 16 / pi / tau  # dipole dislocation density
    # d_min = c_clim * b_s  # the minimum distance between two opposite dislocations to annihilate
    # gamma_dot = rho_m * b_s * nu_0 * exp(-arrhenius * (1 - (tau_eff / tau_sol) ** p) ** q)  # shear rate
    # rho_m = gamma_dot / b_s / lam - 2 * rho_m * gamma_dot / b_s * d_di  # the mobile dislocation density
    # nu_clim = (3 * G * D_0 * Omega_clim / 2 / pi / k_b / T / (d_di - d_min)) * exp(- arrhenius_clim)


    # print(latex(rho_m))
    # print(latex(rho_sum))
    # print(latex(d_di))
    # print(latex(simplify(rho_m)))
