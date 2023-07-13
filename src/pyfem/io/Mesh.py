# -*- coding: utf-8 -*-
"""

"""
from typing import Tuple

from pyfem.utils.visualization import object_slots_to_string


class Mesh:
    __slots__: Tuple = ('type',
                        'file')

    def __init__(self) -> None:
        self.type: str = None  # type: ignore
        self.file: str = None  # type: ignore

    def to_string(self, level: int = 1) -> str:
        return object_slots_to_string(self, level)

    def show(self) -> None:
        print(self.to_string())


if __name__ == "__main__":
    mesh = Mesh()
    mesh.show()
