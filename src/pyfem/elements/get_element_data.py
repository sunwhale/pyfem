from numpy import ndarray

from pyfem.elements.BaseElement import BaseElement
from pyfem.elements.IsoElementShape import IsoElementShape
from pyfem.elements.SolidPlaneSmallStrain import SolidPlaneSmallStrain
from pyfem.io.Dof import Dof
from pyfem.io.Material import Material
from pyfem.io.Section import Section
from pyfem.materials.BaseMaterial import BaseMaterial
from pyfem.utils.colors import error_style

element_data_dict = {
    'SolidPlaneStressSmallStrain': SolidPlaneSmallStrain,
    'SolidPlaneStrainSmallStrain': SolidPlaneSmallStrain
}


def get_element_data(element_id: int,
                     iso_element_shape: IsoElementShape,
                     connectivity: ndarray,
                     node_coords: ndarray,
                     section: Section,
                     dof: Dof,
                     material: Material,
                     material_data: BaseMaterial) -> BaseElement:
    class_name = f'{section.category}{section.type}{section.option}'.strip().replace(' ', '')

    if class_name in element_data_dict:
        return element_data_dict[class_name](element_id=element_id,
                                             iso_element_shape=iso_element_shape,
                                             connectivity=connectivity,
                                             node_coords=node_coords,
                                             section=section,
                                             dof=dof,
                                             material=material,
                                             material_data=material_data)
    else:
        error_msg = f'{class_name} element is not supported.\n'
        error_msg += f'The allowed element types are {list(element_data_dict.keys())}.'
        raise NotImplementedError(error_style(error_msg))


if __name__ == "__main__":
    pass