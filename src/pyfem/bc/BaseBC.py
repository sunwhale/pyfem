from numpy import (empty, ndarray)

from pyfem.io.BC import BC
from pyfem.io.Dof import Dof
from pyfem.mesh.NodeSet import NodeSet
from pyfem.utils.visualization import object_dict_to_string_ndarray


class BaseBC:
    def __init__(self, bc: BC, dof: Dof, nodes: NodeSet) -> None:
        self.bc: BC = bc
        self.dof: Dof = dof
        self.nodes: NodeSet = nodes
        self.bc_node_ids: ndarray = empty(0)
        self.bc_element_ids: ndarray = empty(0)
        self.dof_ids: ndarray = empty(0)
        self.dof_values: ndarray = empty(0)

    def to_string(self, level: int = 1) -> str:
        return object_dict_to_string_ndarray(self, level)

    def show(self) -> None:
        print(self.to_string())


if __name__ == "__main__":
    pass
