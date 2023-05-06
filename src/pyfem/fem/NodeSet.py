import os
from typing import List, Any

import meshio  # type: ignore
import numpy as np

from pyfem.utils.IntKeyDict import IntKeyDict
from pyfem.utils.logger import get_logger

logger = get_logger()


class NodeSet(IntKeyDict):

    def __init__(self):
        super().__init__()
        self.dimension = -1  # int
        self.node_sets = {}  # Dict[str, List[int]]

    def show(self) -> None:
        msg = "Number of nodes ............ %6d\n" % len(self)

        if len(self.node_sets) > 0:
            msg += "  Number of node_sets ........ %6d\n" % len(self.node_sets)
            msg += "  -----------------------------------\n"
            msg += "    name                       #nodes\n"
            msg += "    ---------------------------------\n"

            for name in self.node_sets:
                msg += "    %-16s           %6d \n" % (name, len(self.node_sets[name]))

        logger.info(msg)

    def read_gmsh_file(self, file_name: str) -> None:
        logger.info(f"Reading nodes from {file_name}")

        mesh = meshio.read(file_name, file_format='gmsh')

        # 可以通过以下命令查看支持的单元类型 print(meshio.gmsh.meshio_to_gmsh_type)
        # 可以通过以下命令查看支持的单元类型 print(meshio.gmsh.gmsh_to_meshio_type)

        keywords_1d = ['line']
        keywords_2d = ['triangle', 'quad']
        keywords_3d = ['pyramid', 'hexahedron', 'wedge', 'tetra']

        self.dimension = 2

        # If any 3D mesh type is found, set the dimension to 3
        for cell_name, cell_dict in mesh.cell_sets_dict.items():
            for mesh_type in cell_dict.keys():
                if mesh_type in keywords_3d:
                    self.dimension = 3

        for node_id, coords in enumerate(mesh.points):
            self.add_item_by_id(node_id, coords[:self.dimension])

        for cell_name, cell_dict in mesh.cell_sets_dict.items():
            if cell_name != 'gmsh:bounding_entities':
                for mesh_type, element_ids in cell_dict.items():
                    for element_id in element_ids:
                        cell_nodes = mesh.cells_dict[mesh_type][element_id]
                        for node_id in cell_nodes:
                            self.add_to_sets_by_id(cell_name, node_id)

        for key in self.node_sets:
            self.node_sets[key] = list(set(self.node_sets[key]))

    def add_to_sets_by_id(self, node_set_name: str, node_id: int) -> None:
        if node_set_name not in self.node_sets:
            self.node_sets[node_set_name] = [int(node_id)]
        else:
            self.node_sets[node_set_name].append(int(node_id))

    def get_coords_by_ids(self, node_ids: List[int]) -> np.ndarray[Any, np.dtype[np.float64]]:
        return np.array(self.get_items_by_ids(node_ids))


if __name__ == "__main__":
    from pyfem.utils.logger import set_logger

    set_logger()

    nodes = NodeSet()
    os.chdir(r'F:\Github\pyfem\examples\rectangle')
    # nodes.read_gmsh_file('rectangle.msh')
    nodes.read_gmsh_file('rectangle10000.msh')
    # print(nodes.dimension)
    # print(nodes.node_sets)
    print(nodes.get_coords_by_ids([0, 1, 2]))
    nodes.show()
