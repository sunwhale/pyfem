# -*- coding: utf-8 -*-
"""

"""
from typing import Tuple, Dict

from numpy import eye, ndarray, dot

from pyfem.fem.Timer import Timer
from pyfem.io.Material import Material
from pyfem.io.Section import Section
from pyfem.materials.BaseMaterial import BaseMaterial
from pyfem.utils.colors import error_style


class PhaseFieldDamage(BaseMaterial):

    def __init__(self, material: Material, dimension: int, section: Section) -> None:
        super().__init__(material, dimension, section)
        self.allowed_section_types = ('PlaneStrain',)
        self.create_tangent()

    def create_tangent(self):
        gc = self.material.data[0]
        lc = self.material.data[1]

        if self.section.type in self.allowed_section_types:
            self.tangent = eye(self.dimension) * gc
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
        temperature_gradient = variable['temperature_gradient']
        heat_flux = dot(-self.tangent, temperature_gradient)
        output = {'heat_flux': heat_flux}
        return self.tangent, output


if __name__ == "__main__":
    from pyfem.io.Properties import Properties

    props = Properties()
    props.read_file(r'..\..\..\examples\mechanical_phase\rectangle\Job-1.toml')

    material_data = PhaseFieldDamage(props.materials[0], 3, props.sections[0])
    material_data.show()
