import os
from pathlib import Path
from typing import List, Dict, Union

import meshio  # type: ignore
from numpy import array, ndarray

from pyfem.utils.IntKeyDict import IntKeyDict
from pyfem.utils.logger import get_logger
from pyfem.utils.wrappers import show_running_time
from pyfem.fem.constants import DTYPE

logger = get_logger()


class NodeSet(IntKeyDict):

    def __init__(self) -> None:
        super().__init__()
        self.dimension: int = -1
        self.node_sets: Dict[str, List[int]] = {}

    def show(self) -> None:
        print(self.to_string(0))

    def to_string(self, level=1) -> str:
        msg = 'Number of nodes ............ %6d\n' % len(self)
        space = '   ' * level
        if len(self.node_sets) > 0:
            msg += space + '  Number of node_sets ........ %6d\n' % len(self.node_sets)
            msg += space + '  -----------------------------------\n'
            msg += space + '    name                       #nodes\n'
            msg += space + '    ---------------------------------\n'

            for name in self.node_sets:
                msg += space + '    %-16s           %6d\n' % (name, len(self.node_sets[name]))

        return msg[:-1]

    @show_running_time
    def read_gmsh_file(self, filename: Union[Path, str]) -> None:
        logger.info(f"Reading nodes from {filename}")

        mesh = meshio.read(filename, file_format='gmsh')

        # 可以通过以下命令查看支持的单元类型
        # print(meshio.gmsh.meshio_to_gmsh_type)
        # print(meshio.gmsh.gmsh_to_meshio_type)

        # keywords_1d = ['line']
        # keywords_2d = ['triangle', 'quad']
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

    def read_inp_file(self, file_name: Union[Path, str]) -> None:
        """
        从 ABAQUS inp文件中读取节点信息。
            1. 首先使用meshio.read函数读取文件，获取网格数据。
            2. 然后根据读取到的单元类型确定节点的维度。
            3. 接着遍历节点和单元，将节点添加到节点集合中。
            4. 最后，将节点集合中的重复节点去重。
        """
        logger.info(f"Reading nodes from {file_name}")
        # 从inp文件读取mesh信息
        mesh = meshio.read(file_name, file_format="abaqus")

        # keywords_1d = ['line']
        # keywords_2d = ['triangle', 'quad']
        keywords_3d = ['pyramid', 'hexahedron', 'wedge', 'tetra']

        self.dimension = 2

        # If any 3D mesh type is found, set the dimension to 3
        for cell_name, cell_dict in mesh.cell_sets_dict.items():
            for mesh_type in cell_dict.keys():
                if mesh_type in keywords_3d:
                    self.dimension = 3

        for node_id, coords in enumerate(mesh.points.astype(DTYPE)):
            self.add_item_by_id(node_id, coords[:self.dimension])

        self.node_sets = mesh.point_sets

    def add_to_sets_by_id(self, node_set_name: str, node_id: int) -> None:
        if node_set_name not in self.node_sets:
            self.node_sets[node_set_name] = [int(node_id)]
        else:
            self.node_sets[node_set_name].append(int(node_id))

    def get_coords_by_ids(self, node_ids: List[int]) -> ndarray:
        return array(self.get_items_by_ids(node_ids))


if __name__ == "__main__":
    from pyfem.utils.logger import set_logger

    set_logger()

    nodes = NodeSet()
    # os.chdir(r'/examples/rectangle')
    # # nodes.read_gmsh_file('rectangle100.msh')
    # nodes.read_gmsh_file('rectangle10000.msh')
    # # print(nodes.dimension)
    # # print(nodes.node_sets)
    # print(nodes.get_coords_by_ids([0, 1, 2]))
    # nodes.show()

    os.chdir(r'F:\Github\pyfem\examples\specimen')
    nodes.read_inp_file('Job-1.inp')
    nodes.show()
    print(nodes.dimension)
    print(nodes.node_sets)
    print(nodes.get_coords_by_ids([0, 1, 2]).dtype)
    # nodes.show()
