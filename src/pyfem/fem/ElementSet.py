import os
from typing import List

import meshio
import numpy

from pyfem.utils.IntKeyDict import IntKeyDict
from pyfem.utils.logger import get_logger

logger = get_logger()


class ElementSet(IntKeyDict):

    def __init__(self):
        super().__init__()
        self.element_sets = {}

    def show(self) -> None:
        msg = "Number of elements ......... %6d\n" % len(self)

        if len(self.element_sets) > 0:
            msg += "  Number of element_sets ..... %6d\n" % len(self.element_sets)
            msg += "  -----------------------------------\n"
            msg += "    name                       #elems\n"
            msg += "    ---------------------------------\n"
            for name in self.element_sets:
                msg += "    %-16s           %6d\n" % (name, len(self.element_sets[name]))

        logger.info(msg)

    def get_dof_types(self) -> List[str]:
        dof_types = []
        for element in self:
            for dof_type in element.dof_types:
                if dof_type not in dof_types:
                    dof_types.append(dof_type)
        return dof_types

    def read_gmsh_file(self, file_name: str) -> None:
        mesh = meshio.read(file_name, file_format="gmsh")

        global_element_id = 0
        for cell_name, cell_dict in mesh.cell_sets_dict.items():
            if cell_name != 'gmsh:bounding_entities':
                for mesh_type, element_ids in cell_dict.items():
                    for element_id in element_ids:
                        connectivity = mesh.cells_dict[mesh_type][element_id]
                        self.add_item_by_element_id(global_element_id, cell_name, connectivity)
                        global_element_id += 1

    def add_item_by_element_id(self, element_id: int, element_set_name: str, connectivity: numpy.ndarray[int]) -> None:
        self.add_item_by_id(element_id, connectivity)
        self.add_to_sets_by_id(element_set_name, element_id)

    def add_to_sets_by_id(self, element_set_name: str, element_id: int) -> None:
        if element_set_name not in self.element_sets:
            self.element_sets[element_set_name] = [element_id]
        else:
            self.element_sets[element_set_name].append(element_id)


if __name__ == "__main__":
    from pyfem.utils.logger import set_logger

    set_logger()

    os.chdir(r'F:\Github\pyfem\examples\rectangle')

    elements = ElementSet()
    elements.read_gmsh_file('rectangle.msh')
    # elements.read_gmsh_file('rectangle10000.msh')
    elements.show()
    # print(elements)