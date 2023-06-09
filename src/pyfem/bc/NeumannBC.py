# -*- coding: utf-8 -*-
"""

"""
from typing import Optional

from pyfem.bc.BaseBC import BaseBC
from pyfem.elements.IsoElementShape import IsoElementShape
from pyfem.io.Amplitude import Amplitude
from pyfem.io.BC import BC
from pyfem.io.Dof import Dof
from pyfem.io.Solver import Solver
from pyfem.mesh.MeshData import MeshData

iso_element_shape_dict = {
    'line2': IsoElementShape('line2'),
    'line3': IsoElementShape('line3'),
    'tria3': IsoElementShape('tria3'),
    'tria6': IsoElementShape('tria6'),
    'quad4': IsoElementShape('quad4'),
    'quad8': IsoElementShape('quad8'),
    'tetra4': IsoElementShape('tetra4'),
    'hex8': IsoElementShape('hex8'),
    'hex20': IsoElementShape('hex20')
}


class NeumannBC(BaseBC):
    def __init__(self, bc: BC, dof: Dof, mesh_data: MeshData, solver: Solver, amplitude: Optional[Amplitude]) -> None:
        super().__init__(bc, dof, mesh_data, solver, amplitude)
        self.create_dof_values()

    def create_dof_values(self) -> None:
        bc_elements = self.mesh_data.bc_elements
        nodes = self.mesh_data.nodes
        dimension = self.mesh_data.dimension

        # print(self.mesh_data.elements)
        # print(self.mesh_data.bc_elements)

        bc_element_sets = self.bc.bc_element_sets

        bc_element_ids = []
        # bc_node_ids = []

        # 找到施加面力的边界单元集合

        for bc_element_set in bc_element_sets:
            bc_element_ids += list(self.mesh_data.bc_element_sets[bc_element_set])

        # print(bc_element_ids)
        #
        # for bc_element_id in bc_element_ids:
        #     print(self.mesh_data.bc_elements[bc_element_id])
        #
        # print(iso_element_shape_dict)

        # bc_element_data_list = []
        #
        # for bc_element_id in bc_element_ids:
        #     connectivity = bc_elements[bc_element_id]
        #     node_coords = nodes[connectivity]
        #     iso_element_type = get_iso_element_type(node_coords, dimension - 1)
        #     iso_element_shape = iso_element_shape_dict[iso_element_type]

        # bc_element_data_list[0].show()

        # bc_value = self.bc.value
        # if isinstance(bc_value, float):
        #     iso_element_shape = IsoElementShape('line2')  # 得到二节点线单元的等参单元
        #     gp_weights = iso_element_shape.gp_weights  # 得到高斯点权重
        #     gp_jacobi_dets = self.gp_jacobi_dets  # 得到雅可比矩阵的行列式
        #     gp_number = iso_element_shape.gp_number  # 得到高斯点数
        #     gp_shape_values = iso_element_shape.gp_shape_values
        #     gp_shape_gradients = iso_element_shape.gp_shape_gradients  # 得到等参单元形函数对坐标的导数
        #     gp_coords = iso_element_shape.gp_coords  # 得到高斯点坐标
        #     for i in range(gp_number):
        #         self.dof_values += dot(gp_shape_values[i].transpose(), bc_value[i]) * \
        #                            (sum(dot(gp_coords.transpose(), gp_shape_gradients) ** 2)) ** 0.5 * gp_weights[
        #                                i] * gp_jacobi_dets
        #     self.dof_values = array(self.dof_values)  # 定义函数在SolidPlaneSmallStrain与SolidVolumeSmallStrain中引用


if __name__ == "__main__":
    from pyfem.io.Properties import Properties

    props = Properties()
    props.read_file(r'F:\Github\pyfem\examples\rectangle\rectangle.toml')
    # props.show()

    bc_data = NeumannBC(props.bcs[3], props.dof, props.mesh_data, props.amplitudes[0])
    # bc_data.create_dof_values()
    # bc_data.show()

    # print(det(array([1])))
