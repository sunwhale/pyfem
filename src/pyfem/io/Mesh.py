from typing import Optional

from pyfem.utils.colors import BLUE, END


class Mesh:
    def __init__(self) -> None:
        self.type: Optional[str] = None
        self.file: Optional[str] = None

    def to_string(self, level: int = 1) -> str:
        msg = BLUE + self.__str__() + END
        msg += '\n'
        for key, item in self.__dict__.items():
            msg += '  ' * level + f'|- {key}: {item}\n'
        return msg[:-1]


if __name__ == "__main__":
    mesh = Mesh()
    print(mesh.__dict__.keys())
    print(mesh)
    print(mesh.to_string())
