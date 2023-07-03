# -*- coding: utf-8 -*-
"""

"""
from typing import Tuple, Dict

from numpy import ndarray, empty

from pyfem.fem.Timer import Timer
from pyfem.io.Material import Material
from pyfem.io.Section import Section
from pyfem.utils.visualization import object_dict_to_string_ndarray


class BaseMaterial:
    def __init__(self, material: Material, dimension: int, section: Section) -> None:
        self.material: Material = material
        self.dimension: int = dimension
        self.section: Section = section
        self.allowed_section_types: Tuple = ()
        self.ddsdde: ndarray = empty(0)
        self.output: Dict[str, ndarray] = {}

    def to_string(self, level: int = 1) -> str:
        return object_dict_to_string_ndarray(self, level)

    def show(self) -> None:
        print(self.to_string())

    def get_tangent(self, variable: Dict[str, ndarray],
                    state_variable: Dict[str, ndarray],
                    state_variable_new: Dict[str, ndarray],
                    element_id: int,
                    igp: int,
                    ntens: int,
                    ndi: int,
                    nshr: int,
                    timer: Timer) -> Tuple[ndarray, Dict[str, ndarray]]:
        return self.ddsdde, self.output
