import os
from copy import deepcopy
from typing import List, Any

import meshio  # type: ignore
import numpy as np

from pyfem.io.Section import Section
from pyfem.utils.IntKeyDict import IntKeyDict
from pyfem.utils.logger import get_logger
from pyfem.utils.wrappers import show_running_time

logger = get_logger()


class ElementSet(IntKeyDict):

    def __init__(self):
        super().__init__()
        self.element_sets = {}

    def show(self) -> None:
        logger.info(self.to_string(level=0))

    def to_string(self, level=1) -> str:
        msg = 'Number of elements ......... %6d\n' % len(self)
        space = '   ' * level
        if len(self.element_sets) > 0:
            msg += space + '  Number of element_sets ..... %6d\n' % len(self.element_sets)
            msg += space + '  -----------------------------------\n'
            msg += space + '    name                       #elems\n'
            msg += space + '    ---------------------------------\n'

            for name in self.element_sets:
                msg += space + '    %-16s           %6d\n' % (name, len(self.element_sets[name]))

        return msg[:-1]

    def get_dof_types(self) -> List[str]:
        dof_types = []
        for element in self:
            for dof_type in element.dof_types:
                if dof_type not in dof_types:
                    dof_types.append(dof_type)
        return dof_types

    @show_running_time
    def read_gmsh_file(self, file_name: str, sections: List[Section]) -> None:
        mesh = meshio.read(file_name, file_format="gmsh")

        assigned_element_set = []
        for section in sections:
            assigned_element_set += section.element_sets

        assembly_element_id = 0
        for cell_name, cell_dict in mesh.cell_sets_dict.items():
            if cell_name != 'gmsh:bounding_entities' and cell_name in assigned_element_set:
                for mesh_type, element_ids in cell_dict.items():
                    for element_id in element_ids:
                        connectivity = deepcopy(mesh.cells_dict[mesh_type][element_id])
                        self.add_item_by_element_id(assembly_element_id, cell_name, connectivity)
                        assembly_element_id += 1

    def add_item_by_element_id(self, element_id: int, element_set_name: str,
                               connectivity: np.ndarray[Any, np.dtype[np.int64]]) -> None:
        self.add_item_by_id(element_id, connectivity)
        self.add_to_sets_by_id(element_set_name, element_id)

    def add_to_sets_by_id(self, element_set_name: str, element_id: int) -> None:
        if element_set_name not in self.element_sets:
            self.element_sets[element_set_name] = [element_id]
        else:
            self.element_sets[element_set_name].append(element_id)


if __name__ == "__main__":
    from pyfem.io.Properties import Properties

    props = Properties()
    props.read_file(r'F:\Github\pyfem\examples\rectangle\rectangle.toml')

    os.chdir(r'F:\Github\pyfem\examples\rectangle')

    elements = ElementSet()
    elements.read_gmsh_file('rectangle.msh', props.sections)
    elements.show()
    print(elements.element_sets)
