# -*- coding: utf-8 -*-
"""

"""
from numpy import array

from pyfem.fem.constants import DTYPE
from pyfem.quadrature.BaseQuadrature import BaseQuadrature
from pyfem.utils.colors import error_style


class TetrahedronQuadrature(BaseQuadrature):
    """
    计算三角形积分点的坐标和权重。
    """

    __slots__ = BaseQuadrature.__slots__ + []

    def __init__(self, order: int, dimension: int) -> None:
        dimension = 3
        super().__init__(order, dimension)
        if order == 1:
            third = 1.0 / 3.0
            xi = [[third, third, third]]
            weight = [0.5 * third]
        else:
            raise NotImplementedError(error_style('Only order 1 integration implemented'))

        self.qp_coords = array(xi, dtype=DTYPE)
        self.qp_weights = array(weight, dtype=DTYPE)
        self.qp_number = len(self.qp_weights)


if __name__ == "__main__":
    from pyfem.utils.visualization import print_slots_dict

    print_slots_dict(TetrahedronQuadrature.__slots_dict__)

    quadrature = TetrahedronQuadrature(1, 3)
    quadrature.show()
