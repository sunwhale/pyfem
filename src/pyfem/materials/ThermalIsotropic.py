# -*- coding: utf-8 -*-
"""

"""
from typing import Optional, Tuple, Dict

from numpy import array, eye, diag, ndarray, dot

from pyfem.fem.Timer import Timer
from pyfem.fem.constants import DTYPE
from pyfem.io.Material import Material
from pyfem.materials.BaseMaterial import BaseMaterial
from pyfem.utils.colors import error_style


class ThermalIsotropic(BaseMaterial):
    allowed_option = ['Static', None]

    def __init__(self, material: Material, dimension: int, option: Optional[str] = None) -> None:
        super().__init__(material, dimension, option)
        self.create_tangent()

    def create_tangent(self):
        conductivity = self.material.data[0]
        capacity = self.material.data[1]

        if self.option in self.allowed_option:
            self.ddsdde = eye(self.dimension) * conductivity
        else:
            error_msg = f'{self.option} is not the allowed options {self.allowed_option}'
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
        heat_flux = dot(-self.ddsdde, temperature_gradient)
        output = {'heat_flux': heat_flux}
        return self.ddsdde, output


if __name__ == "__main__":
    from pyfem.io.Properties import Properties

    props = Properties()
    props.read_file(r'F:\Github\pyfem\examples\thermal\1element\hex8\Job-1.toml')

    material_data = ThermalIsotropic(props.materials[0], 3)
    material_data.show()
