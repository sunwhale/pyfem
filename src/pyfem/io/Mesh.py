# -*- coding: utf-8 -*-
"""

"""
from pyfem.utils.visualization import object_dict_to_string


class Mesh:
    def __init__(self) -> None:
        self.type: str = None  # type: ignore
        self.file: str = None  # type: ignore

    def to_string(self, level: int = 1) -> str:
        return object_dict_to_string(self, level)

    def show(self) -> None:
        print(self.to_string())


if __name__ == "__main__":
    mesh = Mesh()
    print(mesh.__dict__.keys())
    print(mesh)
    print(mesh.to_string())
