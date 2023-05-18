from typing import Optional

from pyfem.utils.visualization import object_dict_to_string


class Mesh:
    def __init__(self) -> None:
        self.type: Optional[str] = None
        self.file: Optional[str] = None

    def to_string(self, level: int = 1) -> str:
        return object_dict_to_string(self, level)

    def show(self) -> None:
        print(self.to_string())


if __name__ == "__main__":
    mesh = Mesh()
    print(mesh.__dict__.keys())
    print(mesh)
    print(mesh.to_string())
