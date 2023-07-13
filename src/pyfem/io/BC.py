# -*- coding: utf-8 -*-
"""

"""
from typing import List, Tuple

from pyfem.utils.visualization import object_slots_to_string


class BC:
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
        self.name: str = None  # type: ignore
        self.category: str = None  # type: ignore
        self.type: str = None  # type: ignore
        self.dof: List[str] = None  # type: ignore
        self.node_sets: List[str] = None  # type: ignore
        self.element_sets: List[str] = None  # type: ignore
        self.bc_element_sets: List[str] = None  # type: ignore
        self.value: float = None  # type: ignore
        self.amplitude_name: str = None  # type: ignore

    def to_string(self, level: int = 1) -> str:
        return object_slots_to_string(self, level)

    def show(self) -> None:
        print(self.to_string())


if __name__ == "__main__":
    bc = BC()
    bc.show()
