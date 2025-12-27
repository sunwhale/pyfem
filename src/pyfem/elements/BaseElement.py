# -*- coding: utf-8 -*-
"""

"""
from copy import deepcopy

import numpy as np

from pyfem.fem.Timer import Timer
from pyfem.io.Dof import Dof
from pyfem.io.Material import Material
from pyfem.io.Section import Section
from pyfem.isoelements.IsoElementShape import IsoElementShape
from pyfem.materials.get_material_data import MaterialData
from pyfem.utils.colors import error_style
from pyfem.utils.mechanics import inverse
from pyfem.utils.visualization import object_slots_to_string_ndarray, get_ordinal_number


class BaseElement:
    """
    单元数据实体的基类。

    :ivar element_id: 单元序号
    :vartype element_id: int

    :ivar iso_element_shape: 等参元对象
    :vartype iso_element_shape: IsoElementShape

    :ivar dimension: 单元空间维度
    :vartype dimension: int

    :ivar topological_dimension: 单元拓扑维度
    :vartype topological_dimension: int

    :ivar connectivity: 单元节点序号列表
    :vartype connectivity: np.ndarray

    :ivar node_coords: 单元节点坐标列表
    :vartype node_coords: np.ndarray

    :ivar assembly_conn: 全局单元节点序号列表
    :vartype assembly_conn: np.ndarray

    :ivar dof: io.Dof的自由度对象
    :vartype dof: Dof

    :ivar materials: io.Material的材料对象列表
    :vartype materials: list[Material]

    :ivar section: io.Section的截面对象列表
    :vartype section: list[Section]

    :ivar material_data_list: 材料数据对象列表
    :vartype material_data_list: list[MaterialData]

    :ivar timer: 计时器对象
    :vartype timer: Timer

    :ivar dof_names: 自由度名称列表
    :vartype dof_names: list[str]

    :ivar qp_number: 积分点个数
    :vartype qp_number: int

    :ivar qp_dhdxes: 积分点处的全局坐标形函数梯度
    :vartype qp_dhdxes: np.ndarray(qp_number, 空间维度, 单元节点数)

    :ivar qp_jacobis: 积分点处的雅克比矩阵列表
    :vartype qp_jacobis: np.ndarray(qp_number, 空间维度, 空间维度)

    :ivar qp_jacobi_invs: 积分点处的雅克比矩阵逆矩阵列表
    :vartype qp_jacobi_invs: np.ndarray(qp_number,)

    :ivar qp_jacobi_dets: 积分点处的雅克比矩阵行列式列表
    :vartype qp_jacobi_dets: np.ndarray(qp_number,)

    :ivar qp_weight_times_jacobi_dets: 积分点处的雅克比矩阵行列式乘以积分权重列表
    :vartype qp_weight_times_jacobi_dets: np.ndarray(qp_number,)

    :ivar qp_ddsddes: 积分点处的材料刚度矩阵列表
    :vartype qp_ddsddes: np.ndarray

    :ivar qp_state_variables: 积分点处的状态变量列表
    :vartype qp_state_variables: list[dict[str, np.ndarray]]

    :ivar qp_state_variables_new: 积分点处局部增量时刻的状态变量列表
    :vartype qp_state_variables_new: list[dict[str, np.ndarray]]

    :ivar qp_field_variables: 积分点处场变量字典
    :vartype qp_field_variables: dict[str, np.ndarray]

    :ivar element_dof_number: 单元自由度总数
    :vartype element_dof_number: int

    :ivar element_dof_ids: 单元全局自由度编号列表
    :vartype element_dof_ids: list[int]

    :ivar element_dof_values: 单元全局自由度数值列表
    :vartype element_dof_values: np.ndarray(element_dof_number,)

    :ivar element_ddof_values: 单元全局自由度数值增量列表
    :vartype element_ddof_values: np.ndarray(element_dof_number,)

    :ivar element_fint: 单元内力列表
    :vartype element_fint: np.ndarray(element_dof_number,)

    :ivar element_ftime: 单元时间离散外力列表
    :vartype element_ftime: np.ndarray(element_dof_number,)

    :ivar element_stiffness: 单元刚度矩阵
    :vartype element_stiffness: np.ndarray(element_dof_number, element_dof_number)

    :ivar element_nodal_field_variables: 单元磨平后的场变量字典
    :vartype element_nodal_field_variables: dict[str, np.ndarray]

    :ivar allowed_material_data_list: 许可的单元材料数据类名列表
    :vartype allowed_material_data_list: list[Tuple]

    :ivar allowed_material_number: 许可的单元材料数量
    :vartype allowed_material_number: int
    """

    __slots_dict__: dict = {
        'element_id': ('int', '单元序号'),
        'iso_element_shape': ('IsoElementShape', '等参元对象'),
        'dimension': ('int', '单元空间维度'),
        'topological_dimension': ('int', '单元拓扑维度'),
        'connectivity': ('np.ndarray', '单元节点序号列表'),
        'node_coords': ('np.ndarray', '单元节点坐标列表'),
        'nodes_number': ('np.ndarray', '单元节点坐标列表'),
        'assembly_conn': ('np.ndarray', '全局单元节点序号列表'),
        'dof': ('Dof', 'io.Dof的自由度对象'),
        'materials': ('list[Material]', 'io.Material的材料对象列表'),
        'section': ('list[Section]', 'io.Section的截面对象列表'),
        'material_data_list': ('list[MaterialData]', '材料数据对象列表'),
        'timer': ('Timer', '计时器对象'),
        'dof_names': ('list[str]', '自由度名称列表'),
        'qp_number': ('int', '积分点个数'),
        'qp_dhdxes': ('np.ndarray(qp_number, 空间维度, 单元节点数)', '积分点处的全局坐标形函数梯度'),
        'qp_jacobis': ('np.ndarray(qp_number, 空间维度, 空间维度)', '积分点处的雅克比矩阵列表'),
        'qp_jacobi_invs': ('np.ndarray(qp_number,)', '积分点处的雅克比矩阵逆矩阵列表'),
        'qp_jacobi_dets': ('np.ndarray(qp_number,)', '积分点处的雅克比矩阵行列式列表'),
        'qp_weight_times_jacobi_dets': ('np.ndarray(qp_number,)', '积分点处的雅克比矩阵行列式乘以积分权重列表'),
        'qp_ddsddes': ('np.ndarray', '积分点处的材料刚度矩阵列表'),
        'qp_state_variables': ('list[dict[str, np.ndarray]]', '积分点处的状态变量列表'),
        'qp_state_variables_new': ('list[dict[str, np.ndarray]]', '积分点处局部增量时刻的状态变量列表'),
        'qp_field_variables': ('dict[str, np.ndarray]', '积分点处场变量字典'),
        'element_dof_number': ('int', '单元自由度总数'),
        'element_dof_ids': ('list[int]', '单元全局自由度编号列表'),
        'element_dof_values': ('np.ndarray(element_dof_number,)', '单元全局自由度数值列表'),
        'element_ddof_values': ('np.ndarray(element_dof_number,)', '单元全局自由度数值增量列表'),
        'element_fint': ('np.ndarray(element_dof_number,)', '单元内力列表'),
        'element_ftime': ('np.ndarray(element_dof_number,)', '单元时间离散外力列表'),
        'element_stiffness': ('np.ndarray(element_dof_number, element_dof_number)', '单元刚度矩阵'),
        'element_nodal_field_variables': ('dict[str, np.ndarray]', '单元磨平后的场变量字典'),
        'allowed_material_data_list': ('list[Tuple]', '许可的单元材料数据类名列表'),
        'allowed_material_number': ('int', '许可的单元材料数量')
    }

    __slots__: list = [slot for slot in __slots_dict__.keys()]

    def __init__(self, element_id: int,
                 iso_element_shape: IsoElementShape,
                 connectivity: np.ndarray,
                 node_coords: np.ndarray) -> None:
        self.element_id: int = element_id
        self.iso_element_shape: IsoElementShape = iso_element_shape
        self.dimension: int = node_coords.shape[1]
        self.topological_dimension: int = iso_element_shape.topological_dimension
        self.connectivity: np.ndarray = connectivity
        self.node_coords: np.ndarray = node_coords
        self.nodes_number: int = node_coords.shape[0]
        self.assembly_conn: np.ndarray = None  # type: ignore

        self.dof: Dof = None  # type: ignore
        self.materials: list[Material] = None  # type: ignore
        self.section: Section = None  # type: ignore
        self.material_data_list: list[MaterialData] = None  # type: ignore
        self.timer: Timer = None  # type: ignore

        self.dof_names: list[str] = list()

        self.qp_number: int = self.iso_element_shape.qp_number
        self.qp_dhdxes: np.ndarray = None  # type: ignore
        self.qp_jacobis: np.ndarray = None  # type: ignore
        self.qp_jacobi_invs: np.ndarray = None  # type: ignore
        self.qp_jacobi_dets: np.ndarray = None  # type: ignore
        self.qp_weight_times_jacobi_dets: np.ndarray = None  # type: ignore
        self.qp_ddsddes: list[np.ndarray] = list()
        self.qp_state_variables: list[dict[str, np.ndarray]] = [{} for _ in range(self.qp_number)]
        self.qp_state_variables_new: list[dict[str, np.ndarray]] = [{} for _ in range(self.qp_number)]
        self.qp_field_variables: dict[str, np.ndarray] = dict()
        self.cal_jacobi()

        self.element_dof_number: int = 0
        self.element_dof_ids: list[int] = list()
        self.element_dof_values: np.ndarray = None  # type: ignore
        self.element_ddof_values: np.ndarray = None  # type: ignore
        self.element_fint: np.ndarray = None  # type: ignore
        self.element_ftime: np.ndarray = None  # type: ignore
        self.element_stiffness: np.ndarray = None  # type: ignore
        self.element_nodal_field_variables: dict[str, np.ndarray] = dict()

        self.allowed_material_data_list: list[tuple] = list()
        self.allowed_material_number: int = 0

    def to_string(self, level: int = 1) -> str:
        return object_slots_to_string_ndarray(self, level)

    def show(self) -> None:
        print(self.to_string())

    def cal_jacobi(self) -> None:
        r"""
        计算单元所有积分点处的雅克比矩阵qp_jacobis，雅克比矩阵的逆矩阵qp_jacobi_invs，雅克比矩阵行列式qp_jacobi_dets和雅克比矩阵行列式乘以积分点权重qp_weight_times_jacobi_dets。

        全局坐标系 :math:`\left( {{x_1},{x_2},{x_3}} \right)` 和局部坐标系 :math:`\left( {{\xi _1},{\xi _2},{\xi _3}} \right)` 之间的雅克比矩阵如下：

        .. math::
            \left\{ {\begin{array}{*{20}{c}}
              {{\text{d}}{x_1}} \\
              {{\text{d}}{x_2}} \\
              {{\text{d}}{x_3}}
            \end{array}} \right\} = \left[ {\begin{array}{*{20}{c}}
              {\frac{{\partial {x_1}}}{{\partial {\xi _1}}}}&{\frac{{\partial {x_1}}}{{\partial {\xi _2}}}}&{\frac{{\partial {x_1}}}{{\partial {\xi _3}}}} \\
              {\frac{{\partial {x_2}}}{{\partial {\xi _1}}}}&{\frac{{\partial {x_2}}}{{\partial {\xi _2}}}}&{\frac{{\partial {x_2}}}{{\partial {\xi _3}}}} \\
              {\frac{{\partial {x_3}}}{{\partial {\xi _1}}}}&{\frac{{\partial {x_3}}}{{\partial {\xi _2}}}}&{\frac{{\partial {x_3}}}{{\partial {\xi _3}}}}
            \end{array}} \right]\left\{ {\begin{array}{*{20}{c}}
              {{\text{d}}{\xi _1}} \\
              {{\text{d}}{\xi _2}} \\
              {{\text{d}}{\xi _3}}
            \end{array}} \right\}

        .. math::
            \left[ J \right] = \left[ {\begin{array}{*{20}{c}}
              {\frac{{\partial {x_1}}}{{\partial {\xi _1}}}}&{\frac{{\partial {x_1}}}{{\partial {\xi _2}}}}&{\frac{{\partial {x_1}}}{{\partial {\xi _3}}}} \\
              {\frac{{\partial {x_2}}}{{\partial {\xi _1}}}}&{\frac{{\partial {x_2}}}{{\partial {\xi _2}}}}&{\frac{{\partial {x_2}}}{{\partial {\xi _3}}}} \\
              {\frac{{\partial {x_3}}}{{\partial {\xi _1}}}}&{\frac{{\partial {x_3}}}{{\partial {\xi _2}}}}&{\frac{{\partial {x_3}}}{{\partial {\xi _3}}}}
            \end{array}} \right]

        笛卡尔全局坐标系 :math:`\left( x,y,z \right)` 和局部坐标系 :math:`\left( {\xi ,\eta ,\zeta } \right)` 之间雅克比矩阵可以表示为：

        .. math::
            \left\{ {\begin{array}{*{20}{c}}
              {{\text{d}}x} \\
              {{\text{d}}y} \\
              {{\text{d}}z}
            \end{array}} \right\} = \left[ {\begin{array}{*{20}{c}}
              {\frac{{\partial x}}{{\partial \xi }}}&{\frac{{\partial x}}{{\partial \eta }}}&{\frac{{\partial x}}{{\partial \zeta }}} \\
              {\frac{{\partial y}}{{\partial \xi }}}&{\frac{{\partial y}}{{\partial \eta }}}&{\frac{{\partial y}}{{\partial \zeta }}} \\
              {\frac{{\partial z}}{{\partial \xi }}}&{\frac{{\partial z}}{{\partial \eta }}}&{\frac{{\partial z}}{{\partial \zeta }}}
            \end{array}} \right]\left\{ {\begin{array}{*{20}{c}}
              {{\text{d}}\xi } \\
              {{\text{d}}\eta } \\
              {{\text{d}}\zeta }
            \end{array}} \right\}

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
            \end{array}} \right] = {\left( {\underbrace {\left[ {\begin{array}{*{20}{c}}
              {\frac{{\partial {N_1}}}{{\partial \xi }}}& \cdots &{\frac{{\partial {N_n}}}{{\partial \xi }}} \\
              {\frac{{\partial {N_1}}}{{\partial \eta }}}& \cdots &{\frac{{\partial {N_n}}}{{\partial \eta }}} \\
              {\frac{{\partial {N_1}}}{{\partial \zeta }}}& \cdots &{\frac{{\partial {N_n}}}{{\partial \zeta }}}
            \end{array}} \right]}_{{\text{qp_shape_gradient}}}\underbrace {\left[ {\begin{array}{*{20}{c}}
              {{x_1}}&{{y_1}}&{{z_1}} \\
               \vdots & \vdots & \vdots  \\
              {{x_n}}&{{y_n}}&{{z_n}}
            \end{array}} \right]}_{{\text{node_coords}}}} \right)^T}

        """

        if self.iso_element_shape.qp_shape_gradients.shape[2] == self.node_coords.shape[0]:
            node_coords = self.node_coords
        else:  # 处理内聚力单元
            unique_node_coords, indices = np.unique(self.node_coords, axis=0, return_index=True)
            node_coords = unique_node_coords[indices]

        if self.iso_element_shape.coord_type == 'cartesian':
            # 以下代码为采用numpy高维矩阵乘法的计算方法，计算效率高，但要注意矩阵维度的变化
            # self.qp_jacobis = np.dot(self.iso_element_shape.qp_shape_gradients, node_coords).swapaxes(1, 2)

            # 以下代码为采用numpy爱因斯坦求和约定函数einsum，更简洁明了
            self.qp_jacobis = np.einsum('ijk,kl->ilj', self.iso_element_shape.qp_shape_gradients, node_coords)
            if self.qp_jacobis.shape[1] == self.qp_jacobis.shape[2]:
                self.qp_jacobi_dets = np.linalg.det(self.qp_jacobis)

                # qp_jacobi通常为2×2或3×3的方阵，可以直接根据解析式求逆矩阵，计算效率比numpy.linalg.inv()函数更高
                self.qp_jacobi_invs = inverse(self.qp_jacobis, self.qp_jacobi_dets)
                # 以下代码为采用for循环的计算方法，结构清晰，但计算效率较低
                # qp_dhdxes = []
                # for iqp, (qp_shape_gradient, qp_jacobi_inv) in enumerate(zip(self.iso_element_shape.qp_shape_gradients, self.qp_jacobi_invs)):
                #     qp_dhdxes.append(dot(qp_shape_gradient.transpose(), qp_jacobi_inv).transpose())
                # self.qp_dhdxes = array(qp_dhdxes)
                self.qp_dhdxes = np.einsum('...ij,...ik->...kj', self.iso_element_shape.qp_shape_gradients, self.qp_jacobi_invs)

            elif self.qp_jacobis.shape[1] == 2 and self.qp_jacobis.shape[2] == 1:
                self.qp_jacobi_dets = np.sqrt(np.sum(self.qp_jacobis * self.qp_jacobis, axis=1)).ravel()

            elif self.qp_jacobis.shape[1] == 3 and self.qp_jacobis.shape[2] == 2:
                self.qp_jacobis = np.pad(self.qp_jacobis, pad_width=((0, 0), (0, 0), (0, 1)), mode='constant', constant_values=0)
                self.qp_jacobi_dets = np.zeros(self.qp_jacobis.shape[0])
                for iqp, qp_jacobi in enumerate(self.qp_jacobis):
                    dA = np.zeros(3)
                    dA[0] = np.linalg.norm(np.cross(qp_jacobi[:, 1], qp_jacobi[:, 2]))
                    dA[1] = np.linalg.norm(np.cross(qp_jacobi[:, 2], qp_jacobi[:, 0]))
                    dA[2] = np.linalg.norm(np.cross(qp_jacobi[:, 0], qp_jacobi[:, 1]))
                    self.qp_jacobi_dets[iqp] = np.linalg.norm(dA)

            else:
                raise NotImplementedError(error_style('Unsupported qp_jacobis shape'))

            self.qp_weight_times_jacobi_dets = self.iso_element_shape.qp_weights * self.qp_jacobi_dets

        elif self.iso_element_shape.coord_type == 'barycentric':
            self.qp_jacobis = np.einsum('ijk,kl->ilj', self.iso_element_shape.qp_shape_gradients, node_coords)
            new_rows = np.ones((self.qp_jacobis.shape[0], 1, self.qp_jacobis.shape[2]))
            self.qp_jacobis = np.concatenate((new_rows, self.qp_jacobis), axis=1)

            if self.qp_jacobis.shape[1] == self.qp_jacobis.shape[2]:
                if self.dimension == 2:
                    a = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
                elif self.dimension == 3:
                    a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
                else:
                    raise ValueError(error_style(f'dimension {self.dimension} is not support for the barycentric coordinates'))
                self.qp_jacobi_dets = np.linalg.det(self.qp_jacobis)
                self.qp_jacobi_invs = inverse(self.qp_jacobis, self.qp_jacobi_dets)
                self.qp_jacobi_invs = np.dot(self.qp_jacobi_invs, a)
                self.qp_dhdxes = np.einsum('...ij,...ik->...kj', self.iso_element_shape.qp_shape_gradients, self.qp_jacobi_invs)
            elif self.qp_jacobis.shape[1] == 4 and self.qp_jacobis.shape[2] == 3:
                v1 = node_coords[:, 0] - node_coords[:, 1]
                v2 = node_coords[:, 1] - node_coords[:, 2]
                self.qp_jacobi_dets = np.cross(v1, v2)  # 三角形的面积向量
            else:
                raise NotImplementedError(error_style('Unsupported qp_jacobis shape'))

            if self.topological_dimension == 2:
                self.qp_weight_times_jacobi_dets = self.iso_element_shape.qp_weights * self.qp_jacobi_dets / 2.0
            elif self.topological_dimension == 3:
                self.qp_weight_times_jacobi_dets = self.iso_element_shape.qp_weights * self.qp_jacobi_dets / 6.0
            else:
                raise ValueError(error_style(f'dimension {self.dimension} is not support for the barycentric coordinates'))

    def create_element_dof_ids(self) -> None:
        for node_index in self.assembly_conn:
            for dof_id, _ in enumerate(self.dof_names):
                self.element_dof_ids.append(node_index * len(self.dof_names) + dof_id)

    def check_materials(self) -> None:
        if len(self.materials) != self.allowed_material_number:
            error_msg = f'the length of \'material_names\' of \'{self.section.name}\' -> {type(self).__name__} must be {self.allowed_material_number}, the current \'material_names\' are {self.section.material_names}, please check the .toml file and correct the definition of \'material_names\' of \'{self.section.name}\''
            raise NotImplementedError(error_style(error_msg))
        for i, material_data in enumerate(self.material_data_list):
            material_data_class_name = type(material_data).__name__
            if material_data_class_name not in self.allowed_material_data_list[i]:
                error_msg = f'the \'material_names\' of \'{self.section.name}\' -> {type(self).__name__} are {self.section.material_names}, the {get_ordinal_number(i + 1)} material\'s class is {material_data_class_name}, which is not in the supported list {self.allowed_material_data_list[i]}, please check the .toml file and correct the definition of \'material_names\' of \'{self.section.name}\''
                raise NotImplementedError(error_style(error_msg))

    def create_qp_b_matrices(self) -> None:
        pass

    def update_element_dof_values(self, global_dof_values: np.ndarray) -> None:
        self.element_dof_values = global_dof_values[self.element_dof_ids]

    def update_element_ddof_values(self, global_ddof_values: np.ndarray) -> None:
        self.element_ddof_values = global_ddof_values[self.element_dof_ids]

    def update_element_state_variables(self) -> None:
        self.qp_state_variables = deepcopy(self.qp_state_variables_new)

    def goback_element_state_variables(self) -> None:
        self.qp_state_variables_new = deepcopy(self.qp_state_variables)

    def update_element_material_stiffness_fint(self,
                                               is_update_material: bool = True,
                                               is_update_stiffness: bool = True,
                                               is_update_fint: bool = True, ) -> None:
        pass

    def update_element_field_variables(self) -> None:
        pass


if __name__ == "__main__":
    from pyfem.utils.visualization import print_slots_dict

    print_slots_dict(BaseElement.__slots_dict__)
