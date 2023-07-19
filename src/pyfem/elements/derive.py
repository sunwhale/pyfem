# -*- coding: utf-8 -*-
"""

"""
from sympy import Symbol, diff

if __name__ == '__main__':
    # expr = x**3 + 2*x**2 + x + 1
    #
    # # 对表达式求导
    # derivative = diff(expr, x)
    #
    # # 打印结果
    # print(latex(derivative))

    xi = [Symbol('\\xi_1'), Symbol('\\xi_2')]

    h = [0] * 8

    h[0] = -0.25 * (1.0 - xi[0]) * (1.0 - xi[1]) * (1.0 + xi[0] + xi[1])
    h[1] = -0.25 * (1.0 + xi[0]) * (1.0 - xi[1]) * (1.0 - xi[0] + xi[1])
    h[2] = -0.25 * (1.0 + xi[0]) * (1.0 + xi[1]) * (1.0 - xi[0] - xi[1])
    h[3] = -0.25 * (1.0 - xi[0]) * (1.0 + xi[1]) * (1.0 + xi[0] - xi[1])
    h[4] = 0.5 * (1.0 - xi[0]) * (1.0 + xi[0]) * (1.0 - xi[1])
    h[5] = 0.5 * (1.0 + xi[0]) * (1.0 + xi[1]) * (1.0 - xi[1])
    h[6] = 0.5 * (1.0 - xi[0]) * (1.0 + xi[0]) * (1.0 + xi[1])
    h[7] = 0.5 * (1.0 - xi[0]) * (1.0 + xi[1]) * (1.0 - xi[1])

    for hh in h:
        print(diff(hh, xi[0]))
