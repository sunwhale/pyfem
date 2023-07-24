# -*- coding: utf-8 -*-
"""

"""
from numpy import ones, ndarray

from pyfem.fem.Timer import Timer
from pyfem.io.Material import Material
from pyfem.io.Section import Section
from pyfem.materials.BaseMaterial import BaseMaterial
from pyfem.utils.colors import error_style


class MechanicalThermalExpansion(BaseMaterial):
    """
    热膨胀材料。

    支持的截面属性：('Volume', 'PlaneStress', 'PlaneStrain')

    :ivar alpha: Coefficient of thermal expansion alpha
    :vartype alpha: float
    """

    __slots_dict__: dict = {
        'alpha': ('float', 'Coefficient of thermal expansion alpha')
    }

    __slots__ = BaseMaterial.__slots__ + [slot for slot in __slots_dict__.keys()]

    def __init__(self, material: Material, dimension: int, section: Section) -> None:
        super().__init__(material, dimension, section)
        self.allowed_section_types = ('Volume', 'PlaneStress', 'PlaneStrain')

        self.data_keys = ['Coefficient of thermal expansion alpha']

        if len(self.material.data) != len(self.data_keys):
            raise NotImplementedError(error_style(self.get_data_length_error_msg()))
        else:
            for i, key in enumerate(self.data_keys):
                self.data_dict[key] = material.data[i]

        self.alpha: float = self.data_dict['Coefficient of thermal expansion alpha']

        self.create_tangent()

    def create_tangent(self):
        if self.section.type in self.allowed_section_types:
            if self.dimension == 2:
                self.tangent = ones(3) * self.alpha
                self.tangent[self.dimension:] = 0.0
            elif self.dimension == 3:
                self.tangent = ones(6) * self.alpha
                self.tangent[self.dimension:] = 0.0
        else:
            error_msg = f'{self.section.type} is not the allowed section types {self.allowed_section_types} of the material {type(self).__name__}, please check the definition of the section {self.section.name}'
            raise NotImplementedError(error_style(error_msg))

    def get_tangent(self, variable: dict[str, ndarray],
                    state_variable: dict[str, ndarray],
                    state_variable_new: dict[str, ndarray],
                    element_id: int,
                    igp: int,
                    ntens: int,
                    ndi: int,
                    nshr: int,
                    timer: Timer) -> tuple[ndarray, dict[str, ndarray]]:
        temperature = variable['temperature']
        thermal_strain = -self.tangent * temperature
        output = {'thermal_strain': thermal_strain}
        return self.tangent, output


if __name__ == "__main__":
    from pyfem.utils.visualization import print_slots_dict

    print_slots_dict(MechanicalThermalExpansion.__slots_dict__)

    from pyfem.io.Properties import Properties

    props = Properties()
    props.read_file(r'..\..\..\examples\mechanical_thermal\rectangle\Job-1.toml')
    material_data = MechanicalThermalExpansion(props.materials[2], 3, props.sections[0])
    material_data.show()
