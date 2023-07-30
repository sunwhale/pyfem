# -*- coding: utf-8 -*-
"""

"""
from numpy import array

from pyfem.fem.constants import DTYPE
from pyfem.quadrature.BaseQuadrature import BaseQuadrature
from pyfem.utils.colors import error_style


class TriangleQuadrature(BaseQuadrature):
    """
    计算三角形积分点的坐标和权重。
    """

    __slots__ = BaseQuadrature.__slots__ + []

    def __init__(self, order: int, dimension: int) -> None:
        dimension = 2
        super().__init__(order, dimension)
        if order == 1:
            xi = [[1.0 / 3.0, 1.0 / 3.0]]
            weight = [0.5]

        elif order == 2:
            r1 = 1.0 / 6.0
            r2 = 2.0 / 3.0
            xi = [[r1, r1], [r2, r1], [r1, r2]]
            w1 = 1.0 / 6.0
            weight = [w1, w1, w1]

        elif order == 3:
            r1 = 0.5 * 0.1012865073235
            r2 = 0.5 * 0.7974269853531
            r4 = 0.5 * 0.4701420641051
            r6 = 0.0597158717898
            r7 = 1.0 / 3.0
            xi = [[r1, r1], [r2, r1], [r1, r2], [r4, r6], [r4, r4], [r6, r4], [r7, r7]]
            w1 = 0.1259391805448
            w4 = 0.1323941527885
            w7 = 0.225
            weight = [w1, w1, w1, w4, w4, w4, w7]

        else:
            raise NotImplementedError(error_style('Order must be 1, 3 or 7'))

        self.qp_coords = array(xi, dtype=DTYPE)
        self.qp_weights = array(weight, dtype=DTYPE)
        self.qp_number = len(self.qp_weights)


if __name__ == "__main__":
    from pyfem.utils.visualization import print_slots_dict

    print_slots_dict(TriangleQuadrature.__slots_dict__)

    quadrature = TriangleQuadrature(2, 3)
    quadrature.show()
