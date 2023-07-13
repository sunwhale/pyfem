# -*- coding: utf-8 -*-
"""

"""
from typing import Optional, List, Tuple

from pyfem.io.BaseIO import BaseIO


class Section(BaseIO):
    __slots__: Tuple = ('name',
                        'category',
                        'type',
                        'option',
                        'element_sets',
                        'material_names',
                        'data')

    def __init__(self) -> None:
        super().__init__()
        self.name: str = None  # type: ignore
        self.category: str = None  # type: ignore
        self.type: str = None  # type: ignore
        self.option: Optional[str] = None
        self.element_sets: List[str] = None  # type: ignore
        self.material_names: List[str] = None  # type: ignore
        self.data: List = None  # type: ignore


if __name__ == "__main__":
    section = Section()
    section.show()
