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


class ThermalIsotropic(BaseMaterial):
    __slots__ = BaseMaterial.__slots__ + ('k', 'cp')

    def __init__(self, material: Material, dimension: int, section: Section) -> None:
        super().__init__(material, dimension, section)
        self.allowed_section_types = ('', 'Volume', 'PlaneStress', 'PlaneStrain')

        self.data_keys = ['Conductivity k', 'Capacity cp']

        if len(self.material.data) != len(self.data_keys):
            raise NotImplementedError(error_style(self.get_data_length_error_msg()))
        else:
            for i, key in enumerate(self.data_keys):
                self.data_dict[key] = material.data[i]

        self.k: float = self.data_dict['Conductivity k']
        self.cp: float = self.data_dict['Capacity cp']

        self.create_tangent()

    def create_tangent(self):
        if self.section.type in self.allowed_section_types:
            self.tangent = eye(self.dimension) * self.k
        else:
            raise NotImplementedError(error_style(self.get_section_type_error_msg()))

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
    props.read_file(r'..\..\..\examples\thermal\1element\hex8\Job-1.toml')
    material_data = ThermalIsotropic(props.materials[0], 3, props.sections[0])
    material_data.show()
