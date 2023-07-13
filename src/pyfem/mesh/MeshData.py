# -*- coding: utf-8 -*-
"""

"""
from pathlib import Path
from typing import List, Dict, Tuple, Union

import meshio  # type: ignore
from numpy import ndarray, empty

from pyfem.fem.constants import DTYPE
from pyfem.utils.colors import BLUE, END
from pyfem.utils.colors import error_style, info_style
from pyfem.utils.wrappers import show_running_time


class MeshData:
    __slots__ = ('dimension',
                 'mesh',
                 'nodes',
                 'elements',
                 'bc_elements',
                 'node_sets',
                 'element_sets',
                 'bc_element_sets')

    def __init__(self) -> None:
        self.dimension: int = -1
        self.mesh: meshio.Mesh = None  # type: ignore
        self.nodes: ndarray = empty(0)
        self.elements: List[ndarray] = []
        self.bc_elements: List[ndarray] = []
        self.node_sets: Dict[str, List[int]] = {}
        self.element_sets: Dict[str, List[int]] = {}
        self.bc_element_sets: Dict[str, List[int]] = {}

    def show(self) -> None:
        print(self.to_string(0))

    def to_string(self, level: int = 1) -> str:
        msg = BLUE + self.__str__() + END + '\n'
        space = '   ' * level
        msg += space + info_style('  Number of nodes ............ %6d\n' % len(self.nodes))
        if len(self.node_sets) > 0:
            msg += space + info_style('  Number of node_sets ........ %6d\n' % len(self.node_sets))
            msg += space + '  -----------------------------------\n'
            msg += space + '    name                       #nodes\n'
            msg += space + '    ---------------------------------\n'

            for name in self.node_sets:
                msg += space + '    %-16s           %6d\n' % (name, len(self.node_sets[name]))

        msg += '\n'
        msg += space + info_style('  Number of elements ......... %6d\n' % len(self.elements))
        if len(self.element_sets) > 0:
            msg += space + info_style('  Number of element_sets ..... %6d\n' % len(self.element_sets))
            msg += space + '  -----------------------------------\n'
            msg += space + '    name                       #elems\n'
            msg += space + '    ---------------------------------\n'

            for name in self.element_sets:
                msg += space + '    %-16s           %6d\n' % (name, len(self.element_sets[name]))

        msg += '\n'
        msg += space + info_style('  Number of bc_elements ...... %6d\n' % len(self.bc_elements))
        if len(self.bc_element_sets) > 0:
            msg += space + info_style('  Number of bc_element_sets .. %6d\n' % len(self.bc_element_sets))
            msg += space + '  -----------------------------------\n'
            msg += space + '    name                       #elems\n'
            msg += space + '    ---------------------------------\n'

            for name in self.bc_element_sets:
                msg += space + '    %-16s           %6d\n' % (name, len(self.bc_element_sets[name]))

        return msg

    @show_running_time
    def read_file(self, filename: Union[Path, str], file_format: str = "gmsh") -> None:
        self.mesh = meshio.read(filename, file_format)

        # 单元类型和维度的映射
        cell_type_to_dim = {cell.type: cell.dim for cell in self.mesh.cells}

        self.dimension = max(cell_type_to_dim.values())

        # 建立节点坐标数组
        self.nodes = self.mesh.points[:, :self.dimension].astype(DTYPE)

        # 建立实体单元和边界单元列表
        mesh_type_start_number = {}
        start_number = 0
        bc_mesh_type_start_number = {}
        bc_start_number = 0
        elements = []
        bc_elements = []
        for mesh_type, mesh_element_ids in self.mesh.cells_dict.items():
            if cell_type_to_dim[mesh_type] == self.dimension:
                elements += list(mesh_element_ids)
                mesh_type_start_number[mesh_type] = start_number
                start_number += len(mesh_element_ids)
            elif cell_type_to_dim[mesh_type] == self.dimension - 1:
                bc_elements += list(mesh_element_ids)
                bc_mesh_type_start_number[mesh_type] = start_number
                bc_start_number += len(mesh_element_ids)
            else:
                raise NotImplementedError(error_style(f'unsupported dimension {self.dimension}'))
        self.elements = elements
        self.bc_elements = bc_elements

        # 实体单元和边界单元集合
        for cell_set_name, cell_set_dict in self.mesh.cell_sets_dict.items():
            if cell_set_name != 'gmsh:bounding_entities':
                elements_in_set = []
                bc_elements_in_set = []
                for mesh_type, mesh_element_ids in cell_set_dict.items():
                    if cell_type_to_dim[mesh_type] == self.dimension:
                        elements_in_set += list(mesh_element_ids + mesh_type_start_number[mesh_type])
                    elif cell_type_to_dim[mesh_type] == self.dimension - 1:
                        bc_elements_in_set += list(mesh_element_ids + bc_mesh_type_start_number[mesh_type])
                    else:
                        raise NotImplementedError(error_style(f'unsupported dimension {self.dimension}'))
                if elements_in_set:
                    self.element_sets[cell_set_name] = elements_in_set
                if bc_elements_in_set:
                    self.bc_element_sets[cell_set_name] = bc_elements_in_set

        # 基于边界单元集合建立边界节点集合
        for bc_element_set_name, bc_element_ids in self.bc_element_sets.items():
            for bc_element_id in bc_element_ids:
                self.add_to_node_sets(bc_element_set_name, list(bc_elements[bc_element_id]))

        # 去除边界节点集合中重复的节点
        for key in self.node_sets:
            self.node_sets[key] = list(set(self.node_sets[key]))

        # 添加用户定义的节点集合
        for point_set in self.mesh.point_sets:
            self.node_sets[point_set] = list(self.mesh.point_sets[point_set])

    def add_to_node_sets(self, node_set_name: str, node_ids: List[int]) -> None:
        if node_set_name not in self.node_sets:
            self.node_sets[node_set_name] = node_ids
        else:
            self.node_sets[node_set_name] += node_ids


if __name__ == "__main__":
    mesh_data = MeshData()
    # mesh_data.read_file(r'F:\Github\pyfem\examples\rectangle\quad40000.msh', 'gmsh')
    mesh_data.read_file(r'F:\Github\pyfem\examples\mechanical\rectangle_hole\rectangle_hole_quad4.inp', 'abaqus')
    # mesh_data.read_file(r'F:\Github\pyfem\examples\quad_tria\Job-1.inp', 'abaqus')
    mesh_data.show()
