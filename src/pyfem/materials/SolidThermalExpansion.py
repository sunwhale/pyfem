# -*- coding: utf-8 -*-
"""

"""
from typing import Tuple, Dict

from numpy import ones, ndarray, dot

from pyfem.fem.Timer import Timer
from pyfem.io.Material import Material
from pyfem.io.Section import Section
from pyfem.materials.BaseMaterial import BaseMaterial
from pyfem.utils.colors import error_style


class SolidThermalExpansion(BaseMaterial):

    def __init__(self, material: Material, dimension: int, section: Section) -> None:
        super().__init__(material, dimension, section)
        self.allowed_section_types = ('Static', 'PlaneStrain')
        self.create_tangent()

    def create_tangent(self):
        alpha = self.material.data[0]

        if self.section.type in self.allowed_section_types:
            if self.dimension == 3:
                self.tangent = ones(6) * alpha
                self.tangent[3:] = 0.0
            elif self.dimension == 2:
                self.tangent = ones(3) * alpha
                self.tangent[2:] = 0.0
        else:
            error_msg = f'{self.section.type} is not the allowed section types {self.allowed_section_types} of the material {type(self).__name__}, please check the definition of the section {self.section.name}'
            raise NotImplementedError(error_style(error_msg))

    def get_tangent(self, variable: Dict[str, ndarray],
                    state_variable: Dict[str, ndarray],
                    state_variable_new: Dict[str, ndarray],
                    element_id: int,
                    igp: int,
                    ntens: int,
                    ndi: int,
                    nshr: int,
                    timer: Timer) -> Tuple[ndarray, Dict[str, ndarray]]:
        temperature = variable['temperature']
        thermal_strain = -self.tangent * temperature
        output = {'thermal_strain': thermal_strain}
        return self.tangent, output


if __name__ == "__main__":
    from pyfem.io.Properties import Properties

    props = Properties()
    props.read_file(r'F:\Github\pyfem\examples\thermal\1element\hex8\Job-1.toml')

    material_data = SolidThermalExpansion(props.materials[0], 3, props.sections[0])
    material_data.show()
