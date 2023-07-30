# -*- coding: utf-8 -*-
"""

"""
from numpy import meshgrid, outer, column_stack
from numpy.polynomial.legendre import leggauss

from pyfem.fem.constants import DTYPE
from pyfem.quadrature.BaseQuadrature import BaseQuadrature


class GaussLegendreQuadrature(BaseQuadrature):
    """
    计算 Gauss-Legendre 积分点的坐标和权重。
    """

    __slots__ = BaseQuadrature.__slots__ + []

    def __init__(self, order: int, dimension: int) -> None:
        super().__init__(order, dimension)
        xi, weight = leggauss(order)
        if dimension == 1:
            xi = xi.reshape(len(xi), -1)
            weight = weight.reshape(len(weight), -1)

        elif dimension == 2:
            xi1, xi2 = meshgrid(xi, xi)
            xi1 = xi1.ravel()
            xi2 = xi2.ravel()
            xi = column_stack((xi1, xi2))
            weight = outer(weight, weight)
            weight = weight.ravel()

        elif dimension == 3:
            xi1, xi2, xi3 = meshgrid(xi, xi, xi)
            xi1 = xi1.ravel()
            xi2 = xi2.ravel()
            xi3 = xi3.ravel()
            xi = column_stack((xi1, xi2, xi3))
            weight = outer(outer(weight, weight), weight)
            weight = weight.ravel()

        self.qp_coords = xi.astype(DTYPE)
        self.qp_weights = weight.astype(DTYPE)
        self.qp_number = len(self.qp_weights)


if __name__ == "__main__":
    from pyfem.utils.visualization import print_slots_dict

    print_slots_dict(GaussLegendreQuadrature.__slots_dict__)

    quadrature = GaussLegendreQuadrature(2, 2)
    quadrature.show()
