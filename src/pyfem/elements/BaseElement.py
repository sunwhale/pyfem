# -*- coding: utf-8 -*-
"""

"""
from typing import List, Dict

from numpy import dot, empty, array, ndarray
from numpy.linalg import det, inv

from pyfem.elements.IsoElementShape import IsoElementShape
from pyfem.fem.Timer import Timer
from pyfem.fem.constants import DTYPE
from pyfem.io.Dof import Dof
from pyfem.io.Material import Material
from pyfem.io.Section import Section
from pyfem.materials.BaseMaterial import BaseMaterial
from pyfem.utils.visualization import object_slots_to_string_ndarray


class BaseElement:
    __slots__ = ('element_id', 'iso_element_shape', 'gp_number', 'connectivity', 'assembly_conn', 'node_coords',
                 'gp_jacobis', 'gp_jacobi_invs', 'gp_jacobi_dets', 'gp_weight_times_jacobi_dets', 'dof',
                 'dof_names', 'element_dof_number', 'element_dof_ids', 'element_dof_values', 'element_ddof_values',
                 'element_fint', 'material', 'section', 'material_data', 'timer', 'element_stiffness', 'gp_ddsddes',
                 'gp_state_variables', 'gp_state_variables_new', 'gp_field_variables', 'average_field_variables')

    def __init__(self, element_id: int,
                 iso_element_shape: IsoElementShape,
                 connectivity: ndarray,
                 node_coords: ndarray) -> None:
        self.element_id: int = element_id  # 用户自定义的节点编号
        self.iso_element_shape: IsoElementShape = iso_element_shape
        self.gp_number: int = self.iso_element_shape.gp_number
        self.connectivity: ndarray = connectivity  # 对应用户定义的节点编号
        self.assembly_conn: ndarray = empty(0)  # 对应系统组装时的节点序号
        self.node_coords: ndarray = node_coords
        self.gp_jacobis: ndarray = empty(0, dtype=DTYPE)
        self.gp_jacobi_invs: ndarray = empty(0, dtype=DTYPE)
        self.gp_jacobi_dets: ndarray = empty(0, dtype=DTYPE)
        self.gp_weight_times_jacobi_dets: ndarray = empty(0, dtype=DTYPE)
        self.dof: Dof = None  # type: ignore
        self.dof_names: List[str] = []
        self.element_dof_number: int = 0  # 单元自由度总数
        self.element_dof_ids: List[int] = []  # 对应系统组装时的自由度序号
        self.element_dof_values: ndarray = empty(0, dtype=DTYPE)  # 对应系统组装时的自由度的值
        self.element_ddof_values: ndarray = empty(0, dtype=DTYPE)  # 对应系统组装时的自由度增量的值
        self.element_fint: ndarray = empty(0, dtype=DTYPE)  # 对应系统组装时的内力值
        self.material: Material = None  # type: ignore
        self.section: Section = None  # type: ignore
        self.material_data: BaseMaterial = None  # type: ignore
        self.timer: Timer = None  # type: ignore
        self.element_stiffness: ndarray = empty(0, dtype=DTYPE)
        self.gp_ddsddes: List[ndarray] = []
        self.gp_state_variables: List[Dict[str, ndarray]] = [{} for _ in range(self.iso_element_shape.gp_number)]
        self.gp_state_variables_new: List[Dict[str, ndarray]] = [{} for _ in range(self.iso_element_shape.gp_number)]
        self.gp_field_variables: Dict[str, ndarray] = {}
        self.average_field_variables: Dict[str, ndarray] = {}
        self.cal_jacobi()

    def to_string(self, level: int = 1) -> str:
        return object_slots_to_string_ndarray(self, level)

    def show(self) -> None:
        print(self.to_string())

    def cal_jacobi(self) -> None:
        """
        通过矩阵乘法计算每个积分点上的Jacobi矩阵。
        """

        # 以下代码为采用for循环的计算方法，结构清晰，但计算效率较低
        # self.gp_jacobis = []
        # self.gp_jacobi_invs = []
        # self.gp_jacobi_dets = []
        # for gp_shape_gradient in self.iso_element_shape.gp_shape_gradients:
        #     jacobi = dot(self.node_coords.transpose(), gp_shape_gradient)
        #     self.gp_jacobis.append(jacobi)
        #     self.gp_jacobi_invs.append(inv(jacobi))
        #     self.gp_jacobi_dets.append(det(jacobi))
        # self.gp_jacobis = array(self.gp_jacobis)
        # self.gp_jacobi_invs = array(self.gp_jacobi_invs)
        # self.gp_jacobi_dets = array(self.gp_jacobi_dets)

        # 以下代码为采用numpy高维矩阵乘法的计算方法，计算效率高，但要注意矩阵维度的变化
        self.gp_jacobis = dot(self.node_coords.transpose(), self.iso_element_shape.gp_shape_gradients).swapaxes(0, 1)
        self.gp_jacobi_dets = det(self.gp_jacobis)
        # gp_jacobi通常为2×2或3×3的方阵，可以直接根据解析式求逆矩阵，计算效率比numpy.linalg.inv()函数更高
        # self.gp_jacobi_invs = inv(self.gp_jacobis)
        self.gp_jacobi_invs = inverse(self.gp_jacobis, self.gp_jacobi_dets)
        self.gp_weight_times_jacobi_dets = self.iso_element_shape.gp_weights * self.gp_jacobi_dets

    def create_element_dof_ids(self) -> None:
        for node_index in self.assembly_conn:
            for dof_id, _ in enumerate(self.dof_names):
                self.element_dof_ids.append(node_index * len(self.dof_names) + dof_id)

    def create_gp_b_matrices(self) -> None:
        pass

    def update_element_dof_values(self, global_dof_values: ndarray) -> None:
        self.element_dof_values = global_dof_values[self.element_dof_ids]

    def update_element_ddof_values(self, global_ddof_values: ndarray) -> None:
        self.element_ddof_values = global_ddof_values[self.element_dof_ids]

    def update_element_material_stiffness_fint(self) -> None:
        pass

    def update_material_state(self) -> None:
        pass

    def update_element_stiffness(self) -> None:
        pass

    def update_element_fint(self) -> None:
        pass

    def update_element_state_variables(self) -> None:
        pass

    def update_element_field_variables(self) -> None:
        pass


def inverse(gp_jacobis: ndarray, gp_jacobi_dets: ndarray) -> ndarray:
    """
    对于2×2的矩阵::

        | a11  a12 |
    A = |          |
        | a21  a22 |

    A^-1 = (1 / det(A)) * | a22  -a12 |
                          |           |
                          |-a21   a11 |

    对于3×3的矩阵::
        | a11  a12  a13 |
    A = |               |
        | a21  a22  a23 |
        |               |
        | a31  a32  a33 |

    A^-1 = (1 / det(A)) * |  A22*A33 - A23*A32   A13*A32 - A12*A33   A12*A23 - A13*A22 |
                          |                                                            |
                          |  A23*A31 - A21*A33   A11*A33 - A13*A31   A13*A21 - A11*A23 |
                          |                                                            |
                          |  A21*A32 - A22*A31   A12*A31 - A11*A32   A11*A22 - A12*A21 |
    """
    gp_jacobi_invs = []
    for A, det_A in zip(gp_jacobis, gp_jacobi_dets):
        if A.shape == (2, 2):
            gp_jacobi_invs.append(
                array([[A[1][1], -A[0][1]], [-A[1][0], A[0][0]]]) / det_A)
        elif A.shape == (3, 3):
            gp_jacobi_invs.append(array([[(A[1][1] * A[2][2] - A[1][2] * A[2][1]),
                                          (A[0][2] * A[2][1] - A[0][1] * A[2][2]),
                                          (A[0][1] * A[1][2] - A[0][2] * A[1][1])],
                                         [(A[1][2] * A[2][0] - A[1][0] * A[2][2]),
                                          (A[0][0] * A[2][2] - A[0][2] * A[2][0]),
                                          (A[0][2] * A[1][0] - A[0][0] * A[1][2])],
                                         [(A[1][0] * A[2][1] - A[1][1] * A[2][0]),
                                          (A[0][1] * A[2][0] - A[0][0] * A[2][1]),
                                          (A[0][0] * A[1][1] - A[0][1] * A[1][0])]]) / det_A)
        else:
            return inv(gp_jacobis)
    return array(gp_jacobi_invs)

# def determinate(gp_jacobis: ndarray) -> ndarray:
#     if gp_jacobis.ndim


if __name__ == "__main__":
    pass
    # a = array([[0,0],[0.125,0.125]])
    # inv(a)
