# -*- coding: utf-8 -*-
"""

"""
from sympy import Symbol, diff, simplify, latex


def shape_function_gradient(N: list, xi: list[Symbol]) -> None:
    """
    根据输入的形函数和局部坐标，计算形函数相对局部坐标的梯度表达式，并打印输出。
    """

    for i in range(len(xi)):
        for j in range(len(N)):
            print(f'dhdxi[{i, j}] =', simplify(diff(N[j], xi[i])))


def shape_function_to_latex(N: list, xi: list[Symbol]) -> None:
    """
    根据输入的形函数，打印输出其latex数学表达式，用于注释。
    """

    for i in range(len(N)):
        print('    .. math::')
        print('        N_{', i, '} =', latex(N[i]))
        print()


if __name__ == "__main__":
    # xi = [Symbol('xi[0]'), Symbol('xi[1]'), Symbol('xi[2]')]

    xi = [Symbol('x_0'), Symbol('x_1'), Symbol('x_2')]

    h = [0] * 6

    h[0] = - xi[0] + 2.0 * xi[0] * xi[0]
    h[1] = - xi[1] + 2.0 * xi[1] * xi[1]
    h[2] = - (1.0 - xi[0] - xi[1]) + 2.0 * (1.0 - xi[0] - xi[1]) * (1.0 - xi[0] - xi[1])
    h[3] = 4.0 * xi[0] * xi[1]
    h[4] = 4.0 * xi[1] * (1.0 - xi[0] - xi[1])
    h[5] = 4.0 * xi[0] * (1.0 - xi[0] - xi[1])

    # shape_function_gradient(h, xi[:3])

    # shape_function_to_latex(h, xi[:3])
