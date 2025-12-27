# -*- coding: utf-8 -*-
"""

"""
from typing import Optional

import numpy as np

from pyfem.bc.BaseBC import BaseBC
from pyfem.io.Amplitude import Amplitude
from pyfem.io.BC import BC
from pyfem.io.Dof import Dof
from pyfem.io.Solver import Solver
from pyfem.io.Section import Section
from pyfem.fem.Timer import Timer
from pyfem.isoelements.IsoElementShape import iso_element_shape_dict
from pyfem.isoelements.get_iso_element_type import get_iso_element_type
from pyfem.mesh.MeshData import MeshData
from pyfem.utils.colors import error_style
from pyfem.elements.SurfaceEffect import SurfaceEffect


class NeumannBCTraction(BaseBC):
    r"""
    **Neumann边界条件：压力**

    基于边界条件的属性、自由度属性、网格对象、求解器属性和幅值属性获取系统线性方程组 :math:`{\mathbf{K u}} = {\mathbf{f}}` 中对应等式右边项 :math:`{\mathbf{f}}` 的约束信息。

    Neumann压力边界条件只能施加于边界表面列表 :py:attr:`bc_surface`，其中边界表面列表是由元组（单元编号，单元面名称）对象组成的列表。

    边界表面列表 :py:attr:`bc_surface` 可以由边界条件属性中的节点集合 :py:attr:`pyfem.io.BC.BC.node_sets` 和单元集合 :py:attr:`pyfem.io.BC.BC.element_sets` 通过函数 :py:meth:`get_surface_from_elements_nodes` 确定，也可以由边界条件属性中的边界单元集合 :py:attr:`pyfem.io.BC.BC.bc_element_sets` 通过函数 :py:meth:`get_surface_from_bc_element` 确定。

    对象创建时更新自由度序号列表 :py:attr:`bc_node_ids` 和对应等式右边项取值列表 :py:attr:`bc_fext` 。
    """

    __slots_dict__: dict = {
        'bc_section': ('Section', '边界效应单元截面属性对象'),
    }

    __slots__: list = BaseBC.__slots__ + [slot for slot in __slots_dict__.keys()]

    def __init__(self, bc: BC, dof: Dof, mesh_data: MeshData, solver: Solver, amplitude: Optional[Amplitude]) -> None:
        super().__init__(bc, dof, mesh_data, solver, amplitude)
        self.bc_section: Section = Section()
        self.bc_section.data_dict = {'traction': self.bc.value}
        self.create_dof_values()

    def create_dof_values(self) -> None:
        dimension = self.mesh_data.dimension
        nodes = self.mesh_data.nodes
        elements = self.mesh_data.elements
        bc_elements = self.mesh_data.bc_elements

        node_sets = self.bc.node_sets
        element_sets = self.bc.element_sets
        bc_element_sets = self.bc.bc_element_sets
        bc_value = self.bc.value
        if not (isinstance(bc_value, list)):
            error_msg = f'in {type(self).__name__} \'{self.bc.name}\' the value of \'{bc_value}\' is not a listr'
            raise ValueError(error_style(error_msg))

        if bc_element_sets is not None:
            bc_element_ids = []
            for bc_element_set in bc_element_sets:
                bc_element_ids += list(self.mesh_data.bc_element_sets[bc_element_set])
            for bc_element_id in set(bc_element_ids):
                self.bc_surface += self.get_surface_from_bc_element(bc_element_id, bc_elements[bc_element_id])
        elif element_sets is not None and node_sets is not None:
            element_ids = []
            for element_set in element_sets:
                element_ids += list(self.mesh_data.element_sets[element_set])
            node_ids = []
            for node_set in node_sets:
                node_ids += list(self.mesh_data.node_sets[node_set])
            for element_id in set(element_ids):
                self.bc_surface += self.get_surface_from_elements_nodes(element_id, node_ids)

        bc_dof_ids = []
        bc_fext = []

        for element_id, surface_name in self.bc_surface:
            # 实体单元
            connectivity = elements[element_id]
            node_coords = nodes[connectivity]
            iso_element_type = get_iso_element_type(node_coords)
            iso_element_shape = iso_element_shape_dict[iso_element_type]

            # 边界单元
            bc_connectivity = iso_element_shape.bc_surface_nodes_dict[surface_name]
            bc_node_coords = nodes[bc_connectivity]
            bc_iso_element_type = get_iso_element_type(bc_node_coords, dimension=dimension - 1)
            bc_iso_element_shape = iso_element_shape_dict[bc_iso_element_type]

            bc_element_data = SurfaceEffect(element_id=0,
                                            iso_element_shape=bc_iso_element_shape,
                                            connectivity=bc_connectivity,
                                            node_coords=bc_node_coords,
                                            dof=self.dof,
                                            materials=[],
                                            section=self.bc_section,
                                            material_data_list=[],
                                            timer=Timer())

            element_fext = bc_element_data.get_element_fext()
            bc_assembly_conn = elements[element_id][bc_connectivity]

            bc_element_data.assembly_conn = bc_assembly_conn
            bc_element_data.create_element_dof_ids()
            bc_dof_ids += bc_element_data.element_dof_ids
            bc_fext += list(element_fext)

        self.bc_dof_ids = np.array(bc_dof_ids, dtype='int32')
        self.bc_fext = np.array(bc_fext)


if __name__ == "__main__":
    from pyfem.io.Properties import Properties

    props = Properties()
    props.read_file(r'..\..\..\tests\2elements\hex8.toml')

    for i in range(1, 2):
        print(props.bcs[i].name)
        bc_data = NeumannBCTraction(props.bcs[i], props.dof, props.mesh_data, props.solver, props.amplitudes[0])
        bc_data.show()
