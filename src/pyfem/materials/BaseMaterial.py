from typing import Optional

from numpy import ndarray, dot, empty

from pyfem.io.Material import Material
from pyfem.utils.colors import insert_spaces, BLUE, GREEN, END


class BaseMaterial:
    def __init__(self, material: Material, dimension: int, option: Optional[str] = None) -> None:
        self.material: Material = material
        self.dimension: int = dimension
        self.option: Optional[str] = option
        self.ddsdde: ndarray = empty(0)
        self.state_variables: ndarray = empty(0)

    def to_string(self, level: int = 1) -> str:
        msg = BLUE + self.__str__() + END
        msg += '\n'
        for key, item in self.__dict__.items():
            if isinstance(item, ndarray):
                msg += '  ' * level + GREEN + f'|- {key}: ' + END + f'{type(item)} with shape = {item.shape} \n'
                msg += insert_spaces(5 + (level - 1) * 2, f'{item}') + '\n'
            else:
                msg += '  ' * level + GREEN + f'|- {key}: ' + END + f'{item}\n'
        return msg[:-1]

    def get_stress(self, strain: ndarray) -> ndarray:
        sigma = dot(self.ddsdde, strain)
        return sigma

    def get_tangent(self) -> ndarray:
        return self.ddsdde
