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


class DiffusionIsotropic(BaseMaterial):
    """
    各项同性热传导材料。

    支持的截面属性：('', 'Volume', 'PlaneStress', 'PlaneStrain')

    :ivar d: Diffusion coefficient d
    :vartype d: float
    """

    __slots_dict__: dict = {
        'd': ('float', 'Diffusion coefficient d'),
    }

    __slots__ = BaseMaterial.__slots__ + [slot for slot in __slots_dict__.keys()]

    __data_keys__ = ['Diffusion coefficient d']

    def __init__(self, material: Material, dimension: int, section: Section) -> None:
        super().__init__(material, dimension, section)
        self.allowed_section_types = ('', 'Volume', 'PlaneStress', 'PlaneStrain')

        self.data_keys = self.__data_keys__

        if len(self.material.data) != len(self.data_keys):
            raise NotImplementedError(error_style(self.get_data_length_error_msg()))
        else:
            for i, key in enumerate(self.data_keys):
                self.data_dict[key] = material.data[i]

        self.d: float = self.data_dict['Diffusion coefficient d']

        self.create_tangent()

    def create_tangent(self):
        if self.section.type in self.allowed_section_types:
            self.tangent = eye(self.dimension) * self.d
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

        concentration_gradient = variable['concentration_gradient']
        dconcentration_gradient = variable['dconcentration_gradient']
        if state_variable == {} or timer.time0 == 0.0:
            state_variable['concentration_flux'] = zeros(len(concentration_gradient), dtype=DTYPE)
        concentration_flux = deepcopy(state_variable['concentration_flux'])
        concentration_flux += dot(-self.tangent, dconcentration_gradient)
        state_variable_new['concentration_flux'] = concentration_flux
        output = {'concentration_flux': concentration_flux}
        return self.tangent, output


if __name__ == "__main__":
    from pyfem.utils.visualization import print_slots_dict

    print_slots_dict(DiffusionIsotropic.__slots_dict__)

    from pyfem.io.Properties import Properties

    props = Properties()
    props.read_file(r'..\..\..\examples\diffusion\rectangle\Job-1.toml')
    material_data = DiffusionIsotropic(props.materials[0], 2, props.sections[0])
    material_data.show()
