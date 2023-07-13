# -*- coding: utf-8 -*-
"""

"""
from typing import List, Tuple

from pyfem.io.BaseIO import BaseIO


class BC(BaseIO):
    __slots__: Tuple = ('name',
                        'category',
                        'type',
                        'dof',
                        'node_sets',
                        'element_sets',
                        'bc_element_sets',
                        'value',
                        'amplitude_name')

    def __init__(self) -> None:
        super().__init__()
        self.name: str = None  # type: ignore
        self.category: str = None  # type: ignore
        self.type: str = None  # type: ignore
        self.dof: List[str] = None  # type: ignore
        self.node_sets: List[str] = None  # type: ignore
        self.element_sets: List[str] = None  # type: ignore
        self.bc_element_sets: List[str] = None  # type: ignore
        self.value: float = None  # type: ignore
        self.amplitude_name: str = None  # type: ignore


if __name__ == "__main__":
    bc = BC()
    bc.show()
