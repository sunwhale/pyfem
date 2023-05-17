from typing import List

from numpy import zeros, ones, append, repeat, array
from scipy.sparse import coo_matrix  # type: ignore
from pyfem.utils.wrappers import show_running_time, trace_calls
from pyfem.elements.BaseElement import BaseElement
from pyfem.elements.IsoElementShape import IsoElementShape
from pyfem.io.Dofs import Dofs
from pyfem.fem.NodeSet import NodeSet
from pyfem.fem.ElementSet import ElementSet
from pyfem.materials.PlaneStress import PlaneStress
from pyfem.elements.PlaneStressSmallStrain import PlaneStressSmallStrain


class Assembly:
    def __init__(self, nodes: NodeSet, elements: ElementSet, elements_data: List[BaseElement], dofs: Dofs):
        self.nodes = nodes
        self.elements = elements
        self.elements_data = elements_data
        self.dofs = dofs
        self.global_stiffness = None

    def get_global_stiffness(self):

        global_dofs_name = len(self.nodes) * len(self.dofs.names)

        for element_data in self.elements_data:
            element_data.stiffness


        val = array([], dtype=float)
        row = array([], dtype=int)
        col = array([], dtype=int)


        row = append(row, repeat(el_dofs, len(el_dofs)))

        for i in range(len(el_dofs)):
            col = append(col, el_dofs)

        val = append(val, elemdat.stiff.reshape(len(el_dofs) * len(el_dofs)))

@show_running_time
def main():
    iso_element_shapes = {
        'quad4': IsoElementShape('quad4')
    }
    from pyfem.io.Properties import Properties

    props = Properties()
    props.read_file(r'F:\Github\pyfem\examples\rectangle\rectangle.toml')

    elements = props.elements
    nodes = props.nodes
    dofs = props.dofs

    elements_data = []

    section_of_element_set = {}
    for element_set in elements.element_sets:
        for section in props.sections:
            if element_set in section.element_sets:
                section_of_element_set[element_set] = section

    for element_set_name, element_set in elements.element_sets.items():
        section = props.sections[0]
        material = props.materials[0]
        material_stiffness = PlaneStress(material)
        for element_id in element_set:
            connectivity = elements[element_id]
            if len(connectivity) == 4:
                iso_quad4 = iso_element_shapes['quad4']
                node_coords = array(nodes.get_items_by_ids(list(connectivity)))
                element_data = PlaneStressSmallStrain(iso_element_shape=iso_quad4,
                                                      connectivity=connectivity,
                                                      node_coords=node_coords,
                                                      section=section,
                                                      material=material,
                                                      material_tangent=material_stiffness)
                elements_data.append(element_data)

    assembly = Assembly(nodes, elements, elements_data, dofs)
    assembly.get_global_stiffness()


if __name__ == "__main__":
    main()
