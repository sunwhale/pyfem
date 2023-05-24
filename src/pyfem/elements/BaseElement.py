# -*- coding: utf-8 -*-
"""

"""
from typing import List, Dict,Optional

from numpy import (dot, empty, array, ndarray)
from numpy.linalg import (det, inv)

from pyfem.elements.IsoElementShape import IsoElementShape
from pyfem.io.Dof import Dof
from pyfem.io.Material import Material
from pyfem.io.Section import Section
from pyfem.materials.BaseMaterial import BaseMaterial
from pyfem.utils.visualization import object_dict_to_string_ndarray
from pyfem.utils.wrappers import show_running_time


class BaseElement:
    def __init__(self, element_id: int, iso_element_shape: IsoElementShape, connectivity: ndarray,
                 node_coords: ndarray) -> None:
        self.element_id: int = element_id  # 用户自定义的节点编号
        self.iso_element_shape: IsoElementShape = iso_element_shape
        self.connectivity: ndarray = connectivity  # 对应用户定义的节点编号
        self.assembly_conn: ndarray = empty(0)  # 对应系统组装时的节点序号
        self.node_coords: ndarray = node_coords
        self.gp_jacobis: ndarray = empty(0)
        self.gp_jacobi_invs: ndarray = empty(0)
        self.gp_jacobi_dets: ndarray = empty(0)
        self.dof: Dof = None  # type: ignore
        self.dof_names: List[str] = []
        self.element_dof_ids: List[int] = []  # 对应系统组装时的自由度序号
        self.element_dof_values: ndarray = empty(0)  # 对应系统组装时的自由度的值
        self.element_ddof_values: ndarray = empty(0)  # 对应系统组装时的自由度增量的值
        self.element_dof_number: int = 0  # 单元自由度总数
        self.material: Material = None  # type: ignore
        self.section: Section = None  # type: ignore
        self.material_data: BaseMaterial = None  # type: ignore
        self.stiffness: ndarray = empty(0)
        self.gp_ddsddes: ndarray = empty(0)
        self.gp_state_variables: List[Dict[str, ndarray]] = [{} for _ in range(self.iso_element_shape.nodes_number)]
        self.gp_field_variables: Dict[str, ndarray] = {}
        self.average_field_variables: Dict[str, ndarray] = {}
        self.cal_jacobi()

    def to_string(self, level: int = 1) -> str:
        return object_dict_to_string_ndarray(self, level)

    def show(self) -> None:
        print(self.to_string())

    def cal_jacobi(self) -> None:
        """
        通过矩阵乘法计算每个积分点上的Jacobi矩阵。
        """

        # 以下代码为采用for循环的计算方法，结构清晰，但计算效率较低
        # self.jacobi = []
        # self.jacobi_inv = []
        # self.jacobi_det = []
        # for gp_shape_gradient in self.iso_element_shape.gp_shape_gradients:
        #     jacobi = dot(self.node_coords.transpose(), gp_shape_gradient)
        #     self.jacobi.append(jacobi)
        #     self.jacobi_inv.append(inv(jacobi))
        #     self.jacobi_det.append(det(jacobi))
        # self.jacobi = array(self.jacobi)
        # self.jacobi_inv = array(self.jacobi_inv)
        # self.jacobi_det = array(self.jacobi_det)

        # 以下代码为采用numpy高维矩阵乘法的计算方法，计算效率高，但要注意矩阵维度的变化
        self.gp_jacobis = dot(self.node_coords.transpose(), self.iso_element_shape.gp_shape_gradients).swapaxes(0, 1)
        self.gp_jacobi_invs = inv(self.gp_jacobis)
        self.gp_jacobi_dets = det(self.gp_jacobis)

    def create_element_dof_ids(self) -> None:
        for node_index in self.assembly_conn:
            for dof_id, _ in enumerate(self.dof_names):
                self.element_dof_ids.append(node_index * len(self.dof_names) + dof_id)

    def update_field_variables(self) -> None:
        pass

    def update_element_dof_values(self, solution: ndarray) -> None:
        pass


@show_running_time
def main():
    from pyfem.assembly.Assembly import iso_element_shape_dict
    from pyfem.io.Properties import Properties

    props = Properties()
    props.read_file(r'F:\Github\pyfem\examples\rectangle\rectangle.toml')

    elements = props.elements
    nodes = props.nodes

    base_elements = []

    for element_id, connectivity in elements.items():
        node_coords = array(nodes.get_items_by_ids(list(connectivity)))
        base_element = BaseElement(element_id, iso_element_shape_dict['quad4'], connectivity, node_coords)
        base_elements.append(base_element)

    print(base_elements[0].to_string())


if __name__ == "__main__":
    main()
