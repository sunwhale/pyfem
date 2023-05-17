from typing import List, Dict

from numpy import append, repeat, array
from scipy.sparse import coo_matrix  # type: ignore

from pyfem.elements.BaseElement import BaseElement
from pyfem.elements.IsoElementShape import IsoElementShape
from pyfem.elements.PlaneStressSmallStrain import PlaneStressSmallStrain
from pyfem.io.Properties import Properties
from pyfem.materials.ElasticIsotropic import ElasticIsotropic
from pyfem.utils.wrappers import show_running_time


class Assembly:
    def __init__(self, props: Properties):

        self.element_data_list: List[BaseElement] = []
        self.section_of_element_set: Dict = {}
        self.material_of_section: Dict = {}
        self.materials_dict: Dict = {}
        self.sections_dict: Dict = {}
        self.props = props
        # self.dimension = nodes.dimension
        # self.nodes = nodes
        # self.elements = elements
        #
        # # self.element_sets = elements.element_sets
        # # self.node_sets = nodes.node_sets
        #
        # for connectivity in elements.values():
        #     nodes.get_indices_by_ids(list(connectivity))
        #
        # self.elements_data = elements_data
        # self.dof = dof
        # self.global_stiffness = None

    def init_element_data_list(self):
        elements = self.props.elements
        nodes = self.props.nodes
        sections = self.props.sections
        materials = self.props.materials

        for material in materials:
            self.materials_dict[material.name] = material

        for section in sections:
            self.sections_dict[section.name] = section

        for element_set in elements.element_sets:
            for section in sections:
                if element_set in section.element_sets:
                    self.section_of_element_set[element_set] = section

        iso_quad4 = IsoElementShape('quad4')

        for element_set_name, element_ids in elements.element_sets.items():
            section = self.section_of_element_set[element_set_name]
            material = self.materials_dict[section.material_name]

            material_stiffness = ElasticIsotropic(material)
            for element_id in element_ids:
                connectivity = elements[element_id]
                node_coords = array(nodes.get_items_by_ids(list(connectivity)))
                element_data = PlaneStressSmallStrain(element_id=element_id,
                                                      iso_element_shape=iso_quad4,
                                                      connectivity=connectivity,
                                                      node_coords=node_coords,
                                                      section=section,
                                                      material=material,
                                                      material_data=material_stiffness)
                self.element_data_list.append(element_data)

    def get_global_stiffness(self):

        global_dofs_name = len(self.nodes) * len(self.dof.names)

        val = array([], dtype=float)
        row = array([], dtype=int)
        col = array([], dtype=int)

        for element_data in self.elements_data:
            element_id = element_data.element_id
            element_conn = self.elements[element_id]
            assembly_conn = self.nodes.get_indices_by_ids(list(element_conn))

            element_dof_number = element_data.element_dof_number

            row = append(row, repeat(assembly_conn, element_dof_number))

            for i in range(element_dof_number):
                col = append(col, assembly_conn)

            val = append(val, element_data.stiffness.reshape(element_dof_number * element_dof_number))

        print(row.shape, col.shape, val.shape)


@show_running_time
def main():
    iso_element_shapes = {
        'quad4': IsoElementShape('quad4')
    }
    from pyfem.io.Properties import Properties

    props = Properties()
    props.read_file(r'F:\Github\pyfem\examples\rectangle\rectangle.toml')

    assembly = Assembly(props)

    assembly.init_element_data_list()


if __name__ == "__main__":
    main()
