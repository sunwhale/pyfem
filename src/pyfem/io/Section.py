# -*- coding: utf-8 -*-
"""

"""
from typing import Optional, List

from pyfem.utils.visualization import object_dict_to_string


class Section:
    def __init__(self) -> None:
        self.name: str = None  # type: ignore
        self.category: str = None  # type: ignore
        self.type: str = None  # type: ignore
        self.option: Optional[str] = None
        self.element_sets: List[str] = None  # type: ignore
        self.material_name: str = None  # type: ignore
        self.data: List = None  # type: ignore

    def to_string(self, level: int = 1) -> str:
        return object_dict_to_string(self, level)

    def show(self) -> None:
        print(self.to_string())


if __name__ == "__main__":
    section = Section()
    print(section.__dict__.keys())
    print(section)
    print(section.to_string())
