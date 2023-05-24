from typing import Optional, Dict, Optional

from numpy import ndarray, dot, empty

from pyfem.io.Material import Material
from pyfem.utils.visualization import object_dict_to_string_ndarray


class BaseMaterial:
    def __init__(self, material: Material, dimension: int, option: Optional[str] = None) -> None:
        self.material: Material = material
        self.dimension: int = dimension
        self.option: Optional[str] = option
        self.ddsdde: ndarray = empty(0)

    def to_string(self, level: int = 1) -> str:
        return object_dict_to_string_ndarray(self, level)

    def show(self) -> None:
        print(self.to_string())

    def get_stress(self, strain: ndarray) -> ndarray:
        sigma = dot(self.ddsdde, strain)
        return sigma

    def get_tangent(self, state_variable: Dict[str, ndarray], dstate: ndarray) -> ndarray:
        return self.ddsdde
