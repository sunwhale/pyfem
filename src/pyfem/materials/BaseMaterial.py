from numpy import ndarray, empty

from pyfem.io.Material import Material
from pyfem.utils.colors import insert_spaces, BLUE, GREEN, END


class BaseMaterial:
    def __init__(self):
        self.material = None
        self.ddsdde = None
        self.state_variables = empty(0)

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


if __name__ == "__main__":
    from pyfem.io.Properties import Properties

    props = Properties()
    props.read_file(r'F:\Github\pyfem\examples\rectangle\rectangle.toml')

    base_material = BaseMaterial()
    print(base_material.to_string())
