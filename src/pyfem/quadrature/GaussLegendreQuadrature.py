# -*- coding: utf-8 -*-
"""

"""
import numpy as np

from pyfem.fem.constants import DTYPE
from pyfem.quadrature.BaseQuadrature import BaseQuadrature
from pyfem.utils.colors import error_style


class GaussLegendreQuadrature(BaseQuadrature):
    """
    基于 Gauss-Legendre 多项式计算笛卡尔坐标系下标准线段、正方形和立方体的积分点坐标和权重。

    标准线段::

       (-1)    (0)     (1)
        0---------------1
                +-->ξ

    标准正方形::

        (-1,1)           (1,1)
          3---------------2
          |       η       |
          |       |       |
          |       o--ξ    |
          |               |
          |               |
          0---------------1
        (-1,-1)          (1,-1)

    标准正六面体::

                     (-1,1,1)        (1,1,1)
                      7---------------6
                     /|              /|
                    / |     ζ  η    / |
                   /  |     | /    /  |
        (-1,-1,1) 4---+-----|/----5 (1,-1,1)
                  |   |     o--ξ  |   |
                  |   3-----------+---2 (1,1,-1)
                  |  /(-1,1,-1)   |  /
                  | /             | /
                  |/              |/
                  0---------------1
                 (-1,-1,-1)      (1,-1,-1)

    """

    __slots__ = BaseQuadrature.__slots__ + []

    def __init__(self, order: int, dimension: int) -> None:
        super().__init__(order, dimension)
        xi, weight = np.polynomial.legendre.leggauss(order)
        if dimension == 1:
            xi = xi.reshape(len(xi), -1)
            weight = weight.reshape(len(weight), -1)

        elif dimension == 2:
            xi1, xi2 = np.meshgrid(xi, xi)
            xi1 = xi1.ravel()
            xi2 = xi2.ravel()
            xi = np.column_stack((xi1, xi2))
            weight = np.outer(weight, weight)
            weight = weight.ravel()

        elif dimension == 3:
            xi1, xi2, xi3 = np.meshgrid(xi, xi, xi)
            xi1 = xi1.ravel()
            xi2 = xi2.ravel()
            xi3 = xi3.ravel()
            xi = np.column_stack((xi1, xi2, xi3))
            weight = np.outer(np.outer(weight, weight), weight)
            weight = weight.ravel()

        else:
            raise NotImplementedError(error_style('dimension must be 1, 2 or 3'))

        self.qp_coords = xi.astype(DTYPE)
        self.qp_weights = weight.astype(DTYPE)
        self.qp_number = len(self.qp_weights)


if __name__ == "__main__":
    from pyfem.utils.visualization import print_slots_dict

    print_slots_dict(GaussLegendreQuadrature.__slots_dict__)

    quadrature = GaussLegendreQuadrature(2, 2)
    quadrature.show()
