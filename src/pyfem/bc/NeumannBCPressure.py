# -*- coding: utf-8 -*-
"""

"""
from typing import Optional

from numpy import array, delete, dot, logical_and, ndarray, in1d, all, zeros, sign, cross, sum
from numpy.linalg import det, norm

from pyfem.bc.BaseBC import BaseBC
from pyfem.io.Amplitude import Amplitude
from pyfem.io.BC import BC
from pyfem.io.Dof import Dof
from pyfem.io.Solver import Solver
from pyfem.isoelements.IsoElementShape import iso_element_shape_dict
from pyfem.isoelements.get_iso_element_type import get_iso_element_type
from pyfem.mesh.MeshData import MeshData
from pyfem.utils.colors import error_style


class NeumannBCPressure(BaseBC):
    """
    Neumann边界条件：压力。
    """

    __slots__ = BaseBC.__slots__ + []

    def __init__(self, bc: BC, dof: Dof, mesh_data: MeshData, solver: Solver, amplitude: Optional[Amplitude]) -> None:
        super().__init__(bc, dof, mesh_data, solver, amplitude)
        self.create_dof_values()

    def get_surface_from_bc_element(self, bc_element_id: int, bc_element: ndarray) -> list[tuple[int, str]]:
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
                surface_names = [surface_name for surface_name, nodes_on_surface in
                                 iso_element_shape.nodes_on_surface_dict.items() if
                                 all(nodes_on_surface == nodes_in_element)]
                if len(surface_names) == 1:
                    element_surface.append((element_id, surface_names[0]))
                else:
                    raise NotImplementedError(error_style(f'the surface of element {element_id} is wrong'))

        if len(element_surface) == 1:
            return element_surface
        else:
            raise NotImplementedError(error_style(f'the surface of bc_element {bc_element_id} is wrong'))

    def get_surface_from_elements_nodes(self, element_id: int, node_ids: list[int]) -> list[tuple[int, str]]:
        nodes = self.mesh_data.nodes
        elements = self.mesh_data.elements
        element_surface = []
        nodes_in_element = in1d(elements[element_id], node_ids)
        connectivity = elements[element_id]
        node_coords = nodes[connectivity]
        iso_element_type = get_iso_element_type(node_coords)
        iso_element_shape = iso_element_shape_dict[iso_element_type]
        surface_names = [surface_name for surface_name, nodes_on_surface in
                         iso_element_shape.nodes_on_surface_dict.items() if
                         sum(logical_and(nodes_in_element, nodes_on_surface)) == len(
                             iso_element_shape.bc_surface_nodes_dict[surface_name])]

        for surface_name in surface_names:
            element_surface.append((element_id, surface_name))

        if 1 <= len(element_surface) <= iso_element_shape.bc_surface_number:
            return element_surface
        else:
            raise NotImplementedError(error_style(f'the surface of element {element_id} is wrong'))

    def create_dof_values(self) -> None:
        dimension = self.mesh_data.dimension
        nodes = self.mesh_data.nodes
        elements = self.mesh_data.elements
        bc_elements = self.mesh_data.bc_elements

        node_sets = self.bc.node_sets
        element_sets = self.bc.element_sets
        bc_element_sets = self.bc.bc_element_sets
        bc_value = self.bc.value

        if bc_element_sets is not None:
            bc_element_ids = []
            for bc_element_set in bc_element_sets:
                bc_element_ids += list(self.mesh_data.bc_element_sets[bc_element_set])
            for bc_element_id in set(bc_element_ids):
                self.bc_surface += self.get_surface_from_bc_element(bc_element_id, bc_elements[bc_element_id])
        elif element_sets is not None and node_sets is not None:
            if element_sets == node_sets:
                element_ids = []
                for element_set in element_sets:
                    element_ids += list(self.mesh_data.element_sets[element_set])
                node_ids = []
                for node_set in node_sets:
                    node_ids += list(self.mesh_data.node_sets[node_set])
                for element_id in set(element_ids):
                    self.bc_surface += self.get_surface_from_elements_nodes(element_id, node_ids)
            else:
                raise NotImplementedError(
                    error_style(f'the name of element_sets {element_sets} and node_sets {node_sets} must be the same'))

        bc_dof_ids = []
        bc_fext = []
        bc_dof_names = self.bc.dof
        dof_names = self.dof.names

        for element_id, surface_name in self.bc_surface:
            connectivity = elements[element_id]
            node_coords = nodes[connectivity]
            iso_element_type = get_iso_element_type(node_coords)
            iso_element_shape = iso_element_shape_dict[iso_element_type]

            nodes_number = iso_element_shape.nodes_number
            bc_qp_weights = iso_element_shape.bc_qp_weights
            bc_qp_number = len(bc_qp_weights)
            bc_qp_shape_values = iso_element_shape.bc_qp_shape_values_dict[surface_name]
            bc_qp_shape_gradients = iso_element_shape.bc_qp_shape_gradients_dict[surface_name]
            bc_surface_coord = iso_element_shape.bc_surface_coord_dict[surface_name]
            surface_local_nodes = array(iso_element_shape.bc_surface_nodes_dict[surface_name])
            surface_local_dof_ids = []
            for node_index in surface_local_nodes:
                for _, bc_dof_name in enumerate(bc_dof_names):
                    surface_dof_id = node_index * len(dof_names) + dof_names.index(bc_dof_name)
                    surface_local_dof_ids.append(surface_dof_id)

            surface_nodes = elements[element_id][surface_local_nodes]
            surface_dof_ids = []
            for node_index in surface_nodes:
                for _, bc_dof_name in enumerate(bc_dof_names):
                    surface_dof_id = node_index * len(dof_names) + dof_names.index(bc_dof_name)
                    surface_dof_ids.append(surface_dof_id)

            bc_dof_ids += surface_dof_ids

            element_fext = zeros(nodes_number * len(self.bc.dof))

            surface_norm = {'s1': [[0, 0, 0], [1, 0, 0]],
                            's2': [[0, 0, 0], [0, 1, 0]],
                            's3': [[0, 0, 0], [0, 0, 1]],
                            's4': [[0, 0, 0], [1.0/3.0, 1.0/3.0, 1.0/3.0]]}

            # print(array(surface_norm[surface_name][0]))

            x0, dx0 = iso_element_shape.shape_function(array(surface_norm[surface_name][0]))
            x1, dx1 = iso_element_shape.shape_function(array(surface_norm[surface_name][1]))

            print(x0)
            print(x1)

            print(surface_nodes)
            # print(node_coords)
            surface_nodes_coords = node_coords[surface_nodes]

            a = surface_nodes_coords[0] - surface_nodes_coords[1]
            b = surface_nodes_coords[0] - surface_nodes_coords[2]
            c = cross(a, b)

            print(c)
            # print(bc_norm)
            bc_norm = c / norm(c)
            print(bc_norm)

            for i in range(bc_qp_number):
                bc_qp_jacobi = dot(bc_qp_shape_gradients[i], node_coords).transpose()
                bc_qp_jacobi_sub = delete(bc_qp_jacobi, bc_surface_coord[0], axis=1)
                surface_weight = bc_surface_coord[3]
                if dimension == 2:
                    sigma = -array([[0, bc_value],
                                    [-bc_value, 0]])
                    sigma_times_jacobi = (dot(sigma, bc_qp_jacobi_sub)).transpose()
                    element_fext += (dot(bc_qp_shape_values[i].reshape(1, -1).transpose(), sigma_times_jacobi) *
                                     bc_qp_weights[i] * bc_surface_coord[2] * surface_weight * sign(
                                det(bc_qp_jacobi))).reshape(-1)

                elif dimension == 3:
                    sigma = -bc_value
                    # qp_fext = bc_qp_shape_values[i].transpose() * bc_qp_weights[i] * sigma * surface_weight * det(bc_qp_jacobi)
                    #
                    # for ax, value in enumerate(bc_norm):
                    #     element_dof_ids = [i * len(dof_names) + ax for i in range(nodes_number)]
                    #     element_fext[element_dof_ids] += qp_fext * value

                    if surface_weight == 1:
                        for row in range(bc_qp_jacobi_sub.shape[0]):
                            s = det(delete(bc_qp_jacobi_sub, row, axis=0)) * (-1) ** row
                            qp_fext = (bc_qp_shape_values[i].reshape(1, -1).transpose() * bc_qp_weights[i] * sigma * s *
                                       bc_surface_coord[2] * surface_weight * sign(det(bc_qp_jacobi))).reshape(-1)
                            element_dof_ids = [i * len(dof_names) + row for i in range(nodes_number)]
                            element_fext[element_dof_ids] += qp_fext

                    else:
                        # for ax in range(3):
                        #     bc_qp_jacobi_sub = delete(bc_qp_jacobi, ax, axis=1)
                        #     s = 0
                        #     for row in range(bc_qp_jacobi_sub.shape[0]):
                        #         s += det(delete(bc_qp_jacobi_sub, row, axis=0)) ** 2
                        #     qp_fext = bc_qp_shape_values[i].transpose() * bc_qp_weights[i] * sigma / sqrt(3) * sqrt(
                        #         s) * surface_weight
                        #     element_dof_ids = [i * len(dof_names) + ax for i in range(nodes_number)]
                        #     element_fext[element_dof_ids] += qp_fext

                        qp_fext = bc_qp_shape_values[i].transpose() * bc_qp_weights[i] * sigma * surface_weight * det(
                            bc_qp_jacobi)

                        for ax, value in enumerate(bc_norm):
                            element_dof_ids = [i * len(dof_names) + ax for i in range(nodes_number)]
                            element_fext[element_dof_ids] += qp_fext * value

                else:
                    raise NotImplementedError(
                        error_style(f'dimension {dimension} is not supported of the Neumann boundary condition'))

            surface_fext = []
            for fext in element_fext[surface_local_dof_ids]:
                surface_fext.append(fext)

            bc_fext += list(surface_fext)

        self.bc_dof_ids = array(bc_dof_ids)
        self.bc_fext = array(bc_fext)


if __name__ == "__main__":
    from pyfem.io.Properties import Properties

    # props = Properties()
    # props.read_file(r'..\..\..\examples\mechanical\plane\Job-1.toml')
    # bc_data = NeumannBCPressure(props.bcs[2], props.dof, props.mesh_data, props.solver, None)
    # bc_data.show()

    # props = Properties()
    # props.read_file(r'..\..\..\examples\mechanical\1element\hex8\Job-1.toml')
    # bc_data = NeumannBCPressure(props.bcs[3], props.dof, props.mesh_data, props.solver, None)
    # bc_data.show()

    props = Properties()
    props.read_file(r'..\..\..\examples\mechanical\1element\tetra4\Job-1.toml')
    bc_data = NeumannBCPressure(props.bcs[3], props.dof, props.mesh_data, props.solver, None)
    bc_data.show()
