from pathlib import Path
from typing import List, Dict, Union

import meshio  # type: ignore
from numpy import ndarray, empty

from pyfem.fem.constants import DTYPE
from pyfem.utils.colors import error_style
from pyfem.utils.logger import get_logger
from pyfem.utils.colors import CYAN, MAGENTA, BLUE, END
from pyfem.utils.wrappers import show_running_time


class MeshData:

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
        msg += space + '  Number of nodes ............ %6d\n' % len(self.nodes)
        if len(self.node_sets) > 0:
            msg += space + '  Number of node_sets ........ %6d\n' % len(self.node_sets)
            msg += space + '  -----------------------------------\n'
            msg += space + '    name                       #nodes\n'
            msg += space + '    ---------------------------------\n'

            for name in self.node_sets:
                msg += space + '    %-16s           %6d\n' % (name, len(self.node_sets[name]))

        msg += '\n'
        msg += space + '  Number of elements ......... %6d\n' % len(self.elements)
        if len(self.element_sets) > 0:
            msg += space + '  Number of element_sets ..... %6d\n' % len(self.element_sets)
            msg += space + '  -----------------------------------\n'
            msg += space + '    name                       #elems\n'
            msg += space + '    ---------------------------------\n'

            for name in self.element_sets:
                msg += space + '    %-16s           %6d\n' % (name, len(self.element_sets[name]))

        msg += '\n'
        msg += space + '  Number of bc_elements ...... %6d\n' % len(self.bc_elements)
        if len(self.element_sets) > 0:
            msg += space + '  Number of bc_element_sets .. %6d\n' % len(self.bc_elements)
            msg += space + '  -----------------------------------\n'
            msg += space + '    name                       #elems\n'
            msg += space + '    ---------------------------------\n'

            for name in self.bc_element_sets:
                msg += space + '    %-16s           %6d\n' % (name, len(self.bc_element_sets[name]))

        return msg[:-1]

    @show_running_time
    def read_file(self, filename: Union[Path, str], file_format: str = "gmsh") -> None:
        self.mesh = meshio.read(filename, file_format)

        cell_type_to_dim = {cell.type: cell.dim for cell in self.mesh.cells}

        self.dimension = max(cell_type_to_dim.values())

        self.nodes = self.mesh.points[:, :self.dimension].astype(DTYPE)

        assembly_element_id = 0
        assembly_bc_element_id = 0
        for cell_set_name, cell_set_dict in self.mesh.cell_sets_dict.items():
            if cell_set_name != 'gmsh:bounding_entities':
                for cell_type, element_ids in cell_set_dict.items():
                    cell_conn = self.mesh.cells_dict[cell_type]
                    if cell_type_to_dim[cell_type] == self.dimension:
                        for element_id in element_ids:
                            connectivity = cell_conn[element_id]
                            self.elements.append(connectivity)
                            self.add_to_element_sets(cell_set_name, assembly_element_id)
                            assembly_element_id += 1
                    elif cell_type_to_dim[cell_type] == self.dimension - 1:
                        for element_id in element_ids:
                            connectivity = cell_conn[element_id]
                            self.bc_elements.append(connectivity)
                            self.add_to_bc_element_sets(cell_set_name, assembly_bc_element_id)
                            for node_id in connectivity:
                                self.add_to_node_sets(cell_set_name, node_id)
                            assembly_bc_element_id += 1
                    else:
                        raise NotImplementedError(error_style(f'unsupported dimension {self.dimension}'))

        for key in self.node_sets:
            self.node_sets[key] = list(set(self.node_sets[key]))

        if file_format == 'abaqus':
            self.node_sets = {}
            for point_set in self.mesh.point_sets:
                self.node_sets[point_set] = list(self.mesh.point_sets[point_set])

    def add_to_node_sets(self, node_set_name: str, node_id: int) -> None:
        if node_set_name not in self.node_sets:
            self.node_sets[node_set_name] = [node_id]
        else:
            self.node_sets[node_set_name].append(node_id)

    def add_to_element_sets(self, element_set_name: str, element_id: int) -> None:
        if element_set_name not in self.element_sets:
            self.element_sets[element_set_name] = [element_id]
        else:
            self.element_sets[element_set_name].append(element_id)

    def add_to_bc_element_sets(self, element_set_name: str, element_id: int) -> None:
        if element_set_name not in self.bc_element_sets:
            self.bc_element_sets[element_set_name] = [element_id]
        else:
            self.bc_element_sets[element_set_name].append(element_id)


if __name__ == "__main__":
    mesh_data = MeshData()
    # mesh_data.read_file(r'F:\Github\pyfem\examples\rectangle\quad40000.msh', 'gmsh')
    mesh_data.read_file(r'F:\Github\pyfem\examples\hole\hole_quad4.inp', 'abaqus')
    mesh_data.show()
