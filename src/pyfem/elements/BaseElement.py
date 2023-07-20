# -*- coding: utf-8 -*-
"""

"""
from copy import deepcopy
from typing import Dict, List, Tuple

from numpy import dot, ndarray
from numpy.linalg import det

from pyfem.elements.IsoElementShape import IsoElementShape
from pyfem.fem.Timer import Timer
from pyfem.io.Dof import Dof
from pyfem.io.Material import Material
from pyfem.io.Section import Section
from pyfem.utils.colors import error_style
from pyfem.utils.data_types import MaterialData
from pyfem.utils.mechanics import inverse
from pyfem.utils.visualization import object_slots_to_string_ndarray, get_ordinal_number


class BaseElement:
    """
    单元数据实体的基类。

    :ivar element_id: 单元序号
    :vartype element_id: int

    :ivar iso_element_shape: 等参元对象
    :vartype iso_element_shape: IsoElementShape

    :ivar connectivity: 单元节点序号列表
    :vartype connectivity: ndarray

    :ivar node_coords: 单元节点坐标列表
    :vartype node_coords: ndarray

    :ivar assembly_conn: 全局单元节点序号列表
    :vartype assembly_conn: ndarray

    :ivar dof: io.Dof的自由度对象
    :vartype dof: Dof

    :ivar materials: io.Material的材料对象列表
    :vartype materials: List[Material]

    :ivar section: io.Section的截面对象列表
    :vartype section: List[Section]

    :ivar material_data_list: 材料数据对象列表
    :vartype material_data_list: List[MaterialData]

    :ivar timer: 计时器对象
    :vartype timer: Timer

    :ivar dof_names: 自由度名称列表
    :vartype dof_names: List[str]

    :ivar gp_number: 积分点个数
    :vartype gp_number: int

    :ivar gp_jacobis: 积分点处的雅克比矩阵列表
    :vartype gp_jacobis: ndarray(gp_number, 空间维度, 空间维度)

    :ivar gp_jacobi_invs: 积分点处的雅克比矩阵逆矩阵列表
    :vartype gp_jacobi_invs: ndarray(gp_number,)

    :ivar gp_jacobi_dets: 积分点处的雅克比矩阵行列式列表
    :vartype gp_jacobi_dets: ndarray(gp_number,)

    :ivar gp_weight_times_jacobi_dets: 积分点处的雅克比矩阵行列式乘以积分权重列表
    :vartype gp_weight_times_jacobi_dets: ndarray(gp_number,)

    :ivar gp_ddsddes: 积分点处的材料刚度矩阵列表
    :vartype gp_ddsddes: ndarray

    :ivar gp_state_variables: 积分点处的状态变量列表
    :vartype gp_state_variables: List[Dict[str, ndarray]]

    :ivar gp_state_variables_new: 积分点处局部增量时刻的状态变量列表
    :vartype gp_state_variables_new: List[Dict[str, ndarray]]

    :ivar gp_field_variables: 积分点处场变量字典
    :vartype gp_field_variables: Dict[str, ndarray]

    :ivar element_dof_number: 单元自由度总数
    :vartype element_dof_number: int

    :ivar element_dof_ids: 单元全局自由度编号列表
    :vartype element_dof_ids: List[int]

    :ivar element_dof_values: 单元全局自由度数值列表
    :vartype element_dof_values: ndarray(element_dof_number,)

    :ivar element_ddof_values: 单元全局自由度数值增量列表
    :vartype element_ddof_values: ndarray(element_dof_number,)

    :ivar element_fint: 单元内力列表
    :vartype element_fint: ndarray(element_dof_number,)

    :ivar element_stiffness: 单元刚度矩阵
    :vartype element_stiffness: ndarray(element_dof_number, element_dof_number)

    :ivar element_average_field_variables: 单元磨平后的场变量字典
    :vartype element_average_field_variables: Dict[str, ndarray]

    :ivar allowed_material_data_list: 许可的单元材料数据类名列表
    :vartype allowed_material_data_list: List[Tuple]

    :ivar allowed_material_number: 许可的单元材料数量
    :vartype allowed_material_number: int
    """

    __slots_dict__: Dict = {
        'element_id': ('int', '单元序号'),
        'iso_element_shape': ('IsoElementShape', '等参元对象'),
        'connectivity': ('ndarray', '单元节点序号列表'),
        'node_coords': ('ndarray', '单元节点坐标列表'),
        'assembly_conn': ('ndarray', '全局单元节点序号列表'),
        'dof': ('Dof', 'io.Dof的自由度对象'),
        'materials': ('List[Material]', 'io.Material的材料对象列表'),
        'section': ('List[Section]', 'io.Section的截面对象列表'),
        'material_data_list': ('List[MaterialData]', '材料数据对象列表'),
        'timer': ('Timer', '计时器对象'),
        'dof_names': ('List[str]', '自由度名称列表'),
        'gp_number': ('int', '积分点个数'),
        'gp_jacobis': ('ndarray(gp_number, 空间维度, 空间维度)', '积分点处的雅克比矩阵列表'),
        'gp_jacobi_invs': ('ndarray(gp_number,)', '积分点处的雅克比矩阵逆矩阵列表'),
        'gp_jacobi_dets': ('ndarray(gp_number,)', '积分点处的雅克比矩阵行列式列表'),
        'gp_weight_times_jacobi_dets': ('ndarray(gp_number,)', '积分点处的雅克比矩阵行列式乘以积分权重列表'),
        'gp_ddsddes': ('ndarray', '积分点处的材料刚度矩阵列表'),
        'gp_state_variables': ('List[Dict[str, ndarray]]', '积分点处的状态变量列表'),
        'gp_state_variables_new': ('List[Dict[str, ndarray]]', '积分点处局部增量时刻的状态变量列表'),
        'gp_field_variables': ('Dict[str, ndarray]', '积分点处场变量字典'),
        'element_dof_number': ('int', '单元自由度总数'),
        'element_dof_ids': ('List[int]', '单元全局自由度编号列表'),
        'element_dof_values': ('ndarray(element_dof_number,)', '单元全局自由度数值列表'),
        'element_ddof_values': ('ndarray(element_dof_number,)', '单元全局自由度数值增量列表'),
        'element_fint': ('ndarray(element_dof_number,)', '单元内力列表'),
        'element_stiffness': ('ndarray(element_dof_number, element_dof_number)', '单元刚度矩阵'),
        'element_average_field_variables': ('Dict[str, ndarray]', '单元磨平后的场变量字典'),
        'allowed_material_data_list': ('List[Tuple]', '许可的单元材料数据类名列表'),
        'allowed_material_number': ('int', '许可的单元材料数量')
    }

    __slots__: List = [slot for slot in __slots_dict__.keys()]

    def __init__(self, element_id: int,
                 iso_element_shape: IsoElementShape,
                 connectivity: ndarray,
                 node_coords: ndarray) -> None:
        self.element_id: int = element_id
        self.iso_element_shape: IsoElementShape = iso_element_shape
        self.connectivity: ndarray = connectivity
        self.node_coords: ndarray = node_coords
        self.assembly_conn: ndarray = None  # type: ignore

        self.dof: Dof = None  # type: ignore
        self.materials: List[Material] = None  # type: ignore
        self.section: Section = None  # type: ignore
        self.material_data_list: List[MaterialData] = None  # type: ignore
        self.timer: Timer = None  # type: ignore

        self.dof_names: List[str] = list()

        self.gp_number: int = self.iso_element_shape.gp_number
        self.gp_jacobis: ndarray = None  # type: ignore
        self.gp_jacobi_invs: ndarray = None  # type: ignore
        self.gp_jacobi_dets: ndarray = None  # type: ignore
        self.gp_weight_times_jacobi_dets: ndarray = None  # type: ignore
        self.gp_ddsddes: List[ndarray] = list()
        self.gp_state_variables: List[Dict[str, ndarray]] = [{} for _ in range(self.gp_number)]
        self.gp_state_variables_new: List[Dict[str, ndarray]] = [{} for _ in range(self.gp_number)]
        self.gp_field_variables: Dict[str, ndarray] = dict()
        self.cal_jacobi()

        self.element_dof_number: int = 0
        self.element_dof_ids: List[int] = list()
        self.element_dof_values: ndarray = None  # type: ignore
        self.element_ddof_values: ndarray = None  # type: ignore
        self.element_fint: ndarray = None  # type: ignore
        self.element_stiffness: ndarray = None  # type: ignore
        self.element_average_field_variables: Dict[str, ndarray] = dict()

        self.allowed_material_data_list: List[Tuple] = list()
        self.allowed_material_number: int = 0

    def to_string(self, level: int = 1) -> str:
        return object_slots_to_string_ndarray(self, level)

    def show(self) -> None:
        print(self.to_string())

    def cal_jacobi(self) -> None:
        r"""
        计算单元所有积分点处的雅克比矩阵gp_jacobis，雅克比矩阵的逆矩阵gp_jacobi_invs，雅克比矩阵行列式gp_jacobi_dets和雅克比矩阵行列式乘以积分点权重gp_weight_times_jacobi_dets。

        全局坐标系 :math:`\left( {{x_1},{x_2},{x_3}} \right)` 和局部坐标系 :math:`\left( {{\xi _1},{\xi _2},{\xi _3}} \right)` 之间的雅克比矩阵如下：

        .. math::
            \left[ J \right] = \left[ {\begin{array}{*{20}{c}}
              {\frac{{\partial {x_1}}}{{\partial {\xi _1}}}}&{\frac{{\partial {x_1}}}{{\partial {\xi _2}}}}&{\frac{{\partial {x_1}}}{{\partial {\xi _3}}}} \\
              {\frac{{\partial {x_2}}}{{\partial {\xi _1}}}}&{\frac{{\partial {x_2}}}{{\partial {\xi _2}}}}&{\frac{{\partial {x_2}}}{{\partial {\xi _3}}}} \\
              {\frac{{\partial {x_3}}}{{\partial {\xi _1}}}}&{\frac{{\partial {x_3}}}{{\partial {\xi _2}}}}&{\frac{{\partial {x_3}}}{{\partial {\xi _3}}}}
            \end{array}} \right]

        笛卡尔全局坐标系 :math:`\left( x,y,z \right)` 和局部坐标系 :math:`\left( {\xi ,\eta ,\zeta } \right)` 之间雅克比矩阵可以表示为：

        .. math::
            \left[ J \right] = \left[ {\begin{array}{*{20}{c}}
              {\frac{{\partial x}}{{\partial \xi }}}&{\frac{{\partial x}}{{\partial \eta }}}&{\frac{{\partial x}}{{\partial \zeta }}} \\
              {\frac{{\partial y}}{{\partial \xi }}}&{\frac{{\partial y}}{{\partial \eta }}}&{\frac{{\partial y}}{{\partial \zeta }}} \\
              {\frac{{\partial z}}{{\partial \xi }}}&{\frac{{\partial z}}{{\partial \eta }}}&{\frac{{\partial z}}{{\partial \zeta }}}
            \end{array}} \right]

        采用形函数和单元节点坐标表示：

        .. math::
            \left[ J \right] = \left[ {\begin{array}{*{20}{c}}
              {\sum\limits_{i = 1}^n {\frac{{\partial {N_i}}}{{\partial \xi }}} {x_i}}&{\sum\limits_{i = 1}^n {\frac{{\partial {N_i}}}{{\partial \eta }}} {x_i}}&{\sum\limits_{i = 1}^n {\frac{{\partial {N_i}}}{{\partial \zeta }}} {x_i}} \\
              {\sum\limits_{i = 1}^n {\frac{{\partial {N_i}}}{{\partial \xi }}} {y_i}}&{\sum\limits_{i = 1}^n {\frac{{\partial {N_i}}}{{\partial \eta }}} {y_i}}&{\sum\limits_{i = 1}^n {\frac{{\partial {N_i}}}{{\partial \zeta }}} {y_i}} \\
              {\sum\limits_{i = 1}^n {\frac{{\partial {N_i}}}{{\partial \xi }}} {z_i}}&{\sum\limits_{i = 1}^n {\frac{{\partial {N_i}}}{{\partial \eta }}} {z_i}}&{\sum\limits_{i = 1}^n {\frac{{\partial {N_i}}}{{\partial \zeta }}} {z_i}}
            \end{array}} \right] = \left[ {\begin{array}{*{20}{c}}
              {\frac{{\partial {N_1}}}{{\partial \xi }}}& \cdots &{\frac{{\partial {N_n}}}{{\partial \xi }}} \\
              {\frac{{\partial {N_1}}}{{\partial \eta }}}& \cdots &{\frac{{\partial {N_n}}}{{\partial \eta }}} \\
              {\frac{{\partial {N_1}}}{{\partial \zeta }}}& \cdots &{\frac{{\partial {N_n}}}{{\partial \zeta }}}
            \end{array}} \right]\left[ {\begin{array}{*{20}{c}}
              {{x_1}}&{{y_1}}&{{z_1}} \\
               \vdots & \vdots & \vdots  \\
              {{x_n}}&{{y_n}}&{{z_n}}
            \end{array}} \right]

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
            error_msg = f'the length of \'material_names\' of \'{self.section.name}\' -> {type(self).__name__} must be 3, the current \'material_names\' are {self.section.material_names}, please check the .toml file and correct the definition of \'material_names\' of \'{self.section.name}\''
            raise NotImplementedError(error_style(error_msg))
        for i, material_data in enumerate(self.material_data_list):
            material_data_class_name = type(material_data).__name__
            if material_data_class_name not in self.allowed_material_data_list[i]:
                error_msg = f'the \'material_names\' of \'{self.section.name}\' -> {type(self).__name__} are {self.section.material_names}, the {get_ordinal_number(i + 1)} material\'s class is {material_data_class_name}, which is not in the supported list {self.allowed_material_data_list[i]}, please check the .toml file and correct the definition of \'material_names\' of \'{self.section.name}\''
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


if __name__ == "__main__":
    from pyfem.utils.visualization import print_slots_dict

    print_slots_dict(BaseElement.__slots_dict__)
