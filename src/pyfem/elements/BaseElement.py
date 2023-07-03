# -*- coding: utf-8 -*-
"""

"""
from copy import deepcopy
from typing import List, Dict, Tuple

from numpy import dot, array, ndarray
from numpy.linalg import det, inv

from pyfem.elements.IsoElementShape import IsoElementShape
from pyfem.fem.Timer import Timer
from pyfem.io.Dof import Dof
from pyfem.io.Material import Material
from pyfem.io.Section import Section
from pyfem.materials.BaseMaterial import BaseMaterial
from pyfem.utils.colors import error_style
from pyfem.utils.visualization import object_slots_to_string_ndarray


class BaseElement:
    __slots__ = ('element_id', 'iso_element_shape', 'connectivity', 'node_coords', 'assembly_conn', 'dof', 'materials',
                 'section', 'material_data_list', 'timer', 'dof_names', 'gp_number', 'gp_jacobis', 'gp_jacobi_invs',
                 'gp_jacobi_dets', 'gp_weight_times_jacobi_dets', 'gp_ddsddes', 'gp_state_variables',
                 'gp_state_variables_new', 'gp_field_variables', 'element_dof_number', 'element_dof_ids',
                 'element_dof_values', 'element_ddof_values', 'element_fint', 'element_stiffness',
                 'element_average_field_variables', 'allowed_material_data_list', 'allowed_material_number')

    def __init__(self, element_id: int,
                 iso_element_shape: IsoElementShape,
                 connectivity: ndarray,
                 node_coords: ndarray) -> None:
        self.element_id: int = element_id  # 用户自定义的节点编号
        self.iso_element_shape: IsoElementShape = iso_element_shape
        self.connectivity: ndarray = connectivity  # 对应用户定义的节点编号
        self.node_coords: ndarray = node_coords
        self.assembly_conn: ndarray = None  # type: ignore  # 对应系统组装时的节点序号

        self.dof: Dof = None  # type: ignore
        self.materials: List[Material] = None  # type: ignore
        self.section: Section = None  # type: ignore
        self.material_data_list: List[BaseMaterial] = None  # type: ignore
        self.timer: Timer = None  # type: ignore

        self.dof_names: List[str] = []

        self.gp_number: int = self.iso_element_shape.gp_number
        self.gp_jacobis: ndarray = None  # type: ignore
        self.gp_jacobi_invs: ndarray = None  # type: ignore
        self.gp_jacobi_dets: ndarray = None  # type: ignore
        self.gp_weight_times_jacobi_dets: ndarray = None  # type: ignore
        self.gp_ddsddes: List[ndarray] = []
        self.gp_state_variables: List[Dict[str, ndarray]] = [{} for _ in range(self.gp_number)]
        self.gp_state_variables_new: List[Dict[str, ndarray]] = [{} for _ in range(self.gp_number)]
        self.gp_field_variables: Dict[str, ndarray] = {}
        self.cal_jacobi()

        self.element_dof_number: int = 0  # 单元自由度总数
        self.element_dof_ids: List[int] = []  # 对应系统组装时的自由度序号
        self.element_dof_values: ndarray = None  # type: ignore  # 对应系统组装时的自由度的值
        self.element_ddof_values: ndarray = None  # type: ignore  # 对应系统组装时的自由度增量的值
        self.element_fint: ndarray = None  # type: ignore  # 对应系统组装时的内力值
        self.element_stiffness: ndarray = None  # type: ignore
        self.element_average_field_variables: Dict[str, ndarray] = {}

        self.allowed_material_data_list: Tuple = ()
        self.allowed_material_number: int = 1

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
        #     jacobi = dot(gp_shape_gradient, self.node_coords).transpose()
        #     self.gp_jacobis.append(jacobi)
        #     self.gp_jacobi_invs.append(inv(jacobi))
        #     self.gp_jacobi_dets.append(det(jacobi))
        # self.gp_jacobis = array(self.gp_jacobis)
        # self.gp_jacobi_invs = array(self.gp_jacobi_invs)
        # self.gp_jacobi_dets = array(self.gp_jacobi_dets)

        # 以下代码为采用numpy高维矩阵乘法的计算方法，计算效率高，但要注意矩阵维度的变化
        self.gp_jacobis = dot(self.iso_element_shape.gp_shape_gradients, self.node_coords).swapaxes(1, 2)
        self.gp_jacobi_dets = det(self.gp_jacobis)

        # gp_jacobi通常为2×2或3×3的方阵，可以直接根据解析式求逆矩阵，计算效率比numpy.linalg.inv()函数更高
        self.gp_jacobi_invs = inverse(self.gp_jacobis, self.gp_jacobi_dets)
        self.gp_weight_times_jacobi_dets = self.iso_element_shape.gp_weights * self.gp_jacobi_dets

    def create_element_dof_ids(self) -> None:
        for node_index in self.assembly_conn:
            for dof_id, _ in enumerate(self.dof_names):
                self.element_dof_ids.append(node_index * len(self.dof_names) + dof_id)

    def check_materials(self) -> None:
        if len(self.materials) != self.allowed_material_number:
            error_msg = f'{type(self).__name__} section supports only {self.allowed_material_number} thermal material, please check the definition of {self.section.name}, length of {self.section.material_names} must be {self.allowed_material_number}'
            raise NotImplementedError(error_style(error_msg))
        for material_data in self.material_data_list:
            material_data_class_name = type(material_data).__name__
            if material_data_class_name not in self.allowed_material_data_list:
                error_msg = f'{material_data_class_name} is not the supported material of {type(self).__name__} section, the allowed materials are {self.allowed_material_data_list}'
                raise NotImplementedError(error_style(error_msg))

    def create_gp_b_matrices(self) -> None:
        pass

    def update_element_dof_values(self, global_dof_values: ndarray) -> None:
        self.element_dof_values = global_dof_values[self.element_dof_ids]

    def update_element_ddof_values(self, global_ddof_values: ndarray) -> None:
        self.element_ddof_values = global_ddof_values[self.element_dof_ids]

    def update_element_state_variables(self) -> None:
        self.gp_state_variables = deepcopy(self.gp_state_variables_new)

    def update_element_material_stiffness_fint(self) -> None:
        pass

    def update_material_state(self) -> None:
        pass

    def update_element_stiffness(self) -> None:
        pass

    def update_element_fint(self) -> None:
        pass

    def update_element_field_variables(self) -> None:
        pass


def inverse(gp_jacobis: ndarray, gp_jacobi_dets: ndarray) -> ndarray:
    """
    对于2×2和3×3的矩阵求逆直接带入下面的公式，其余的情况则调用np.linalg.inv()函数

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


if __name__ == "__main__":
    pass
