# -*- coding: utf-8 -*-
"""

"""
import numpy as np

from pyfem.utils.visualization import object_slots_to_string_ndarray


class BaseQuadrature:
    """
    数值积分基类。

    :ivar dimension: 空间维度
    :vartype dimension: int

    :ivar order: 插值阶次
    :vartype order: int

    :ivar qp_coords: 积分点坐标
    :vartype qp_coords: np.ndarray

    :ivar qp_weights: 积分点权重
    :vartype qp_weights: np.ndarray
    """

    __slots_dict__: dict = {
        'order': ('int', '插值阶次'),
        'dimension': ('int', '空间维度'),
        'qp_number': ('int', '积分点数量'),
        'qp_coords': ('np.ndarray(qp_number, dimension)', '积分点坐标'),
        'qp_weights': ('np.ndarray(qp_number, dimension)', '积分点权重'),
    }

    __slots__: list = [slot for slot in __slots_dict__.keys()]

    def __init__(self, order: int, dimension: int) -> None:
        self.order: int = order
        self.dimension: int = dimension
        self.qp_number: int = 0
        self.qp_coords: np.ndarray = np.zeros(0)
        self.qp_weights: np.ndarray = np.zeros(0)

    def to_string(self, level: int = 1) -> str:
        return object_slots_to_string_ndarray(self, level)

    def show(self) -> None:
        print(self.to_string())

    def get_quadrature_coords_and_weights(self) -> tuple[np.ndarray, np.ndarray]:
        return self.qp_coords, self.qp_weights


if __name__ == "__main__":
    from pyfem.utils.visualization import print_slots_dict

    print_slots_dict(BaseQuadrature.__slots_dict__)

    quadrature = BaseQuadrature(1, 3)
    quadrature.show()
