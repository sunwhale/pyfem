# -*- coding: utf-8 -*-
"""

"""
from typing import Tuple

from pyfem.io.BaseIO import BaseIO


class Mesh(BaseIO):
    __slots__: Tuple = ('type',
                        'file')

    def __init__(self) -> None:
        super().__init__()
        self.type: str = None  # type: ignore
        self.file: str = None  # type: ignore


if __name__ == "__main__":
    mesh = Mesh()
    mesh.show()
