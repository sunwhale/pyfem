# -*- coding: utf-8 -*-
"""

"""
from typing import Dict, List, Tuple

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
        self.tangent: ndarray = empty(0)
        self.output: Dict[str, ndarray] = {}
        self.data_keys: List[str] = []
        self.data_dict: Dict[str, float] = {}

    def get_section_type_error_msg(self) -> str:
        return f'\'{self.section.type}\' is not the allowed section types {self.allowed_section_types} of the material \'{self.material.name}\' -> {type(self).__name__}, please check the definition of the section \'{self.section.name}\''

    def get_data_length_error_msg(self) -> str:
        return f'the length of \'data\' -> {self.material.data} of \'{self.material.name}\' -> {type(self).__name__} must be {len(self.data_keys)} and stored in the order of {self.data_keys}'

    def to_string(self, level: int = 1) -> str:
        return object_dict_to_string_ndarray(self, level)

    def show(self) -> None:
        print(self.to_string())

    def create_tangent(self) -> None:
        pass

    def get_tangent(self, variable: Dict[str, ndarray],
                    state_variable: Dict[str, ndarray],
                    state_variable_new: Dict[str, ndarray],
                    element_id: int,
                    igp: int,
                    ntens: int,
                    ndi: int,
                    nshr: int,
                    timer: Timer) -> Tuple[ndarray, Dict[str, ndarray]]:
        return self.tangent, self.output
