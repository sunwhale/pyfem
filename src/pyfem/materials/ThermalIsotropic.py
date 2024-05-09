# -*- coding: utf-8 -*-
"""

"""
from copy import deepcopy

from numpy import eye, ndarray, dot, zeros

from pyfem.fem.Timer import Timer
from pyfem.fem.constants import DTYPE
from pyfem.io.Material import Material
from pyfem.io.Section import Section
from pyfem.materials.BaseMaterial import BaseMaterial
from pyfem.utils.colors import error_style


class ThermalIsotropic(BaseMaterial):
    """
    各项同性热传导材料。

    支持的截面属性：('', 'Volume', 'PlaneStress', 'PlaneStrain')

    :ivar k: Conductivity k
    :vartype k: float

    :ivar cp: Capacity cp
    :vartype cp: float
    """

    __slots_dict__: dict = {
        'k': ('float', 'Conductivity k'),
        'cp': ('float', 'Capacity cp'),
    }

    __slots__ = BaseMaterial.__slots__ + [slot for slot in __slots_dict__.keys()]

    __data_keys__ = ['Conductivity k', 'Capacity cp']

    def __init__(self, material: Material, dimension: int, section: Section) -> None:
        super().__init__(material, dimension, section)
        self.allowed_section_types = ('', 'Volume', 'PlaneStress', 'PlaneStrain')

        self.data_keys = self.__data_keys__

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

    def get_tangent(self, variable: dict[str, ndarray],
                    state_variable: dict[str, ndarray],
                    state_variable_new: dict[str, ndarray],
                    element_id: int,
                    iqp: int,
                    ntens: int,
                    ndi: int,
                    nshr: int,
                    timer: Timer) -> tuple[ndarray, dict[str, ndarray]]:
        # 全量格式
        # temperature_gradient = variable['temperature_gradient']
        # heat_flux = dot(-self.tangent, temperature_gradient)

        # 增量格式
        temperature_gradient = variable['temperature_gradient']
        dtemperature_gradient = variable['dtemperature_gradient']
        if state_variable == {} or timer.time0 == 0.0:
            state_variable['heat_flux'] = zeros(len(temperature_gradient), dtype=DTYPE)
        heat_flux = deepcopy(state_variable['heat_flux'])
        heat_flux += dot(-self.tangent, dtemperature_gradient)
        state_variable_new['heat_flux'] = heat_flux

        output = {'heat_flux': heat_flux}
        return self.tangent, output


if __name__ == "__main__":
    from pyfem.utils.visualization import print_slots_dict

    print_slots_dict(ThermalIsotropic.__slots_dict__)

    from pyfem.io.Properties import Properties

    props = Properties()
    props.read_file(r'..\..\..\examples\thermal\1element\hex8\Job-1.toml')
    material_data = ThermalIsotropic(props.materials[0], 3, props.sections[0])
    material_data.show()
