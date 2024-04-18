# -*- coding: utf-8 -*-
"""

"""
from typing import Optional

from numpy import array, delete, dot, logical_and, ndarray, in1d, all, zeros, sign, cross, sum, sqrt
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
    r"""
    **Neumann边界条件：压力**

    基于边界条件的属性、自由度属性、网格对象、求解器属性和幅值属性获取系统线性方程组 :math:`{\mathbf{K u}} = {\mathbf{f}}` 中对应等式右边项 :math:`{\mathbf{f}}` 的约束信息。

    Neumann压力边界条件只能施加于边界表面列表 :py:attr:`bc_surface`，其中边界表面列表是由元组（单元编号，单元面名称）对象组成的列表。

    边界表面列表 :py:attr:`bc_surface` 可以由边界条件属性中的节点集合 :py:attr:`pyfem.io.BC.BC.node_sets` 和单元集合 :py:attr:`pyfem.io.BC.BC.element_sets` 通过函数 :py:meth:`get_surface_from_elements_nodes` 确定，也可以由边界条件属性中的边界单元集合 :py:attr:`pyfem.io.BC.BC.bc_element_sets` 通过函数 :py:meth:`get_surface_from_bc_element` 确定。

    对象创建时更新自由度序号列表 :py:attr:`bc_node_ids` 和对应等式右边项取值列表 :py:attr:`bc_fext` 。

    ========================================
    理论-基于等参元的计算方法
    ========================================

    ----------------------------------------
    1. 基于变分法的基本公式
    ----------------------------------------

    考虑在单元边界上施加某个分布载荷的情况，对于二维问题，面力 :math:`\mathbf{\bar p}` 在 :math:`\left( m \right)` 号单元所产生的等效节点载荷为：

    .. math::
        {{\mathbf{R}}^{\left( m \right)}} = \int\limits_{s_p^{\left( m \right)}} {{{\left( {{{\mathbf{N}}^{\left( m \right)}}} \right)}^{\text{T}}}{\mathbf{\bar p}}{\text{d}}s}

    对于三维问题，面力 :math:`\mathbf{\bar p}` 在 :math:`\left( m \right)` 号单元所产生的等效节点载荷为：

    .. math::
        {{\mathbf{R}}^{\left( m \right)}} = \iint\limits_{S_p^{\left( m \right)}} {{{\left( {{{\mathbf{N}}^{\left( m \right)}}} \right)}^{\text{T}}}{\mathbf{\bar p}}{\text{d}}S}

    若单元的某个面上只作用着沿外法线方向的法向载荷

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
        if not (isinstance(bc_value, float) or isinstance(bc_value, int)):
            error_msg = f'in {type(self).__name__} \'{self.bc.name}\' the value of \'{bc_value}\' is not a float or int number'
            raise NotImplementedError(error_style(error_msg))

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

            for i in range(bc_qp_number):
                bc_qp_jacobi = dot(bc_qp_shape_gradients[i], node_coords).transpose()  # 此处雅克比矩阵的行列式为体积比
                bc_qp_jacobi_sub = delete(bc_qp_jacobi, bc_surface_coord[0], axis=1)
                surface_weight = bc_surface_coord[3]
                if dimension == 2:
                    sigma = -array([[0.0, bc_value],
                                    [-bc_value, 0.0]])
                    sigma_times_jacobi = (dot(sigma, bc_qp_jacobi_sub)).transpose()
                    element_fext += (dot(bc_qp_shape_values[i].reshape(1, -1).transpose(), sigma_times_jacobi) *
                                     bc_qp_weights[i] * bc_surface_coord[2] * surface_weight * sign(det(bc_qp_jacobi))).reshape(-1)

                elif dimension == 3:
                    sigma = -bc_value  # type: ignore
                    if surface_weight == 1:
                        for row in range(bc_qp_jacobi_sub.shape[0]):
                            s = det(delete(bc_qp_jacobi_sub, row, axis=0)) * (-1) ** row
                            qp_fext = (bc_qp_shape_values[i].reshape(1, -1).transpose() * bc_qp_weights[i] * sigma * s *
                                       bc_surface_coord[2] * surface_weight * sign(det(bc_qp_jacobi))).reshape(-1)
                            element_dof_ids = [i * len(dof_names) + row for i in range(nodes_number)]
                            element_fext[element_dof_ids] += qp_fext

                    else:
                        # 第二类曲面积分的方法目前难以统一处理斜边上的积分，因此简化为第一类曲面积分，通过节点坐标计算面的外法线方向
                        surface_nodes_coords = nodes[surface_nodes]

                        a = surface_nodes_coords[0] - surface_nodes_coords[1]
                        b = surface_nodes_coords[0] - surface_nodes_coords[2]
                        c = cross(a, b)

                        bc_norm = c / norm(c)
                        surface_weight = norm(c)  # 三角形面积的2倍

                        qp_fext = bc_qp_shape_values[i].transpose() * bc_qp_weights[i] * sigma * surface_weight

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

        self.bc_dof_ids = array(bc_dof_ids, dtype='int32')
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

    # props = Properties()
    # props.read_file(r'..\..\..\examples\mechanical\1element\quad4\Job-1.toml')
    # bc_data = NeumannBCPressure(props.bcs[2], props.dof, props.mesh_data, props.solver, None)
    # bc_data.show()
