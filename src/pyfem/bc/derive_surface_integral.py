# -*- coding: utf-8 -*-
"""

"""
from sympy import Symbol, diff, simplify, latex, expand


if __name__ == "__main__":
    dx_dxi = Symbol(r'\frac{\partial x}{\partial \xi}')
    dy_dxi = Symbol(r'\frac{\partial y}{\partial \xi}')
    dz_dxi = Symbol(r'\frac{\partial z}{\partial \xi}')

    dx_deta = Symbol(r'\frac{\partial x}{\partial \eta}')
    dy_deta = Symbol(r'\frac{\partial y}{\partial \eta}')
    dz_deta = Symbol(r'\frac{\partial z}{\partial \eta}')

    dx_dzeta = Symbol(r'\frac{\partial x}{\partial \zeta}')
    dy_dzeta = Symbol(r'\frac{\partial y}{\partial \zeta}')
    dz_dzeta = Symbol(r'\frac{\partial z}{\partial \zeta}')

    dxi = Symbol(r'\text{d} \xi')
    deta = Symbol(r'\text{d} \eta')
    dzeta = Symbol(r'\text{d} \zeta')

    dx = dx_dxi * dxi + dx_deta * deta + dx_dzeta * dzeta
    dy = dy_dxi * dxi + dy_deta * deta + dy_dzeta * dzeta
    dz = dz_dxi * dxi + dz_deta * deta + dz_dzeta * dzeta

    dS = ((dy * dz) ** 2 + (dz * dx) ** 2 + (dx * dy) ** 2) ** 0.5

    print(latex(expand((dy * dz) ** 2)))
