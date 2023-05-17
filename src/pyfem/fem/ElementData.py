from typing import List, Any, Union

from pyfem.utils.IntKeyDict import IntKeyDict
from pyfem.elements.BaseElement import BaseElement
from pyfem.utils.colors import error_style


class ElementDataList(list):

    def __init__(self):
        super().__init__()

    # def __setitem__(self, key: int, value: BaseElement) -> None:
    #     if isinstance(value, BaseElement):
    #         if key in self:
    #             error_msg = f'Key {key} already exists'
    #             raise KeyError(error_style(error_msg))
    #         else:
    #             super().__setitem__(key, value)
    #     else:
    #         error_msg = f'item of {type(self)} must be an object of BaseElement'
    #         raise TypeError(error_style(error_msg))

    def append(self, value: BaseElement) -> None:
        if isinstance(value, BaseElement):
            super().append(value)
        else:
            error_msg = f'only BaseElement object can append to {type(self)}'
            raise TypeError(error_style(error_msg))


if __name__ == "__main__":
    from numpy import empty, zeros, dot, array, ndarray
    from pyfem.io.Properties import Properties
    from pyfem.elements.BaseElement import BaseElement
    from pyfem.elements.IsoElementShape import IsoElementShape
    from pyfem.io.Material import Material
    from pyfem.io.Section import Section
    from pyfem.materials.PlaneStress import PlaneStress
    from pyfem.utils.wrappers import show_running_time, trace_calls
    from pyfem.elements.PlaneStressSmallStrain import PlaneStressSmallStrain

    iso_element_shapes = {
        'quad4': IsoElementShape('quad4'),
        'line2': IsoElementShape('line2')
    }

    props = Properties()
    props.read_file(r'F:\Github\pyfem\examples\rectangle\rectangle.toml')

    elements = props.elements
    nodes = props.nodes

    element_data_list = []

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
                element_data_list.append(element_data)

    print(element_data_list[0])

    element_data_list.__getitem__(0)