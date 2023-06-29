# -*- coding: utf-8 -*-
"""

"""
from numpy import delete, dot, ndarray, in1d, all, sqrt, zeros
from typing import Dict, Optional, Tuple

from pyfem.bc.BaseBC import BaseBC
from pyfem.elements.IsoElementShape import IsoElementShape
from pyfem.elements.get_iso_element_type import get_iso_element_type
from pyfem.io.Amplitude import Amplitude
from pyfem.io.BC import BC
from pyfem.io.Dof import Dof
from pyfem.io.Solver import Solver
from pyfem.mesh.MeshData import MeshData
from pyfem.utils.colors import error_style

iso_element_shape_dict: Dict[str, IsoElementShape] = {
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


class NeumannBCPressure(BaseBC):
    def __init__(self, bc: BC, dof: Dof, mesh_data: MeshData, solver: Solver, amplitude: Optional[Amplitude]) -> None:
        super().__init__(bc, dof, mesh_data, solver, amplitude)
        self.create_dof_values()

    def get_surface_from_bc_element(self, bc_element_id: int, bc_element: ndarray) -> Tuple[int, str]:
        nodes = self.mesh_data.nodes
        elements = self.mesh_data.elements
        element_surface = []
        for element_id, element in enumerate(elements):
            is_element_surface = all(in1d(bc_element, element))
            if is_element_surface:
                nodes_in_element = in1d(element, bc_element)
                connectivity = elements[element_id]
                node_coords = nodes[connectivity]
                iso_element_type = get_iso_element_type(node_coords)
                iso_element_shape = iso_element_shape_dict[iso_element_type]
                surface_names = [key for key, item in iso_element_shape.nodes_to_surface_dict.items() if all(item == nodes_in_element)]
                if len(surface_names) == 1:
                    element_surface.append((element_id, surface_names[0]))
                else:
                    raise NotImplementedError(error_style(f'the surface of element {element_id} is wrong'))

        if len(element_surface) == 1:
            return element_surface[0]
        else:
            raise NotImplementedError(error_style(f'the surface of bc_element {bc_element_id} is wrong'))

    def create_dof_values(self) -> None:
        dimension = self.mesh_data.dimension
        nodes = self.mesh_data.nodes
        elements = self.mesh_data.elements
        bc_elements = self.mesh_data.bc_elements
        bc_element_sets = self.bc.bc_element_sets

        bc_element_ids = []
        for bc_element_set in bc_element_sets:
            bc_element_ids += list(self.mesh_data.bc_element_sets[bc_element_set])

        for bc_element_id in set(bc_element_ids):
            self.bc_surface.append(self.get_surface_from_bc_element(bc_element_id, bc_elements[bc_element_id]))

        bc_value = self.bc.value

        for element_id, surface_name in self.bc_surface:
            connectivity = elements[element_id]
            node_coords = nodes[connectivity]
            iso_element_type = get_iso_element_type(node_coords)
            iso_element_shape = iso_element_shape_dict[iso_element_type]

            nodes_number = iso_element_shape.nodes_number
            bc_gp_weights = iso_element_shape.bc_gp_weights
            bc_gp_number = len(bc_gp_weights)
            bc_gp_shape_values = iso_element_shape.bc_gp_shape_values_dict[surface_name]
            bc_gp_shape_gradients = iso_element_shape.bc_gp_shape_gradients_dict[surface_name]
            bc_surface_coord = iso_element_shape.bc_surface_coord_dict[surface_name]

            bc_fext = zeros(nodes_number)

            for i in range(bc_gp_number):
                bc_gp_jacobi = dot(bc_gp_shape_gradients[i], node_coords).transpose()
                bc_gp_jacobi_sub = delete(bc_gp_jacobi, bc_surface_coord[0], axis=1)
                bc_fext += bc_gp_shape_values[i].transpose() * bc_value * sqrt(sum(bc_gp_jacobi_sub**2))

            print(bc_fext)

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

    bc_data = NeumannBCPressure(props.bcs[3], props.dof, props.mesh_data, props.solver, props.amplitudes[0])
    # bc_data.create_dof_values()
    bc_data.show()

    # print(det(array([1])))
