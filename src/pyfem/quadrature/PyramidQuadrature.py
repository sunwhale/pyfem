# -*- coding: utf-8 -*-
"""

"""
import numpy as np

from pyfem.fem.constants import DTYPE
from pyfem.quadrature.BaseQuadrature import BaseQuadrature
from pyfem.utils.colors import error_style


class PyramidQuadrature(BaseQuadrature):
    """
    计算笛卡尔坐标系下标准金字塔的积分点坐标和权重。
    """

    __slots__ = BaseQuadrature.__slots__ + []

    def __init__(self, order: int, dimension: int) -> None:
        dimension = 3
        super().__init__(order, dimension)
        if order == 1:  # order 1, qp_number 1
            qp_coords_and_weights = np.array([[0., 0., -0.5, 128.0 / 27.0]], dtype=DTYPE)
        else:
            raise NotImplementedError(error_style('order must be 1'))

        self.qp_coords = qp_coords_and_weights[:, 0:2]
        self.qp_weights = qp_coords_and_weights[:, 2:3]
        self.qp_number = len(self.qp_weights)


if __name__ == "__main__":
    from pyfem.utils.visualization import print_slots_dict

    print_slots_dict(PyramidQuadrature.__slots_dict__)

    quadrature = PyramidQuadrature(1, 2)
    quadrature.show()
