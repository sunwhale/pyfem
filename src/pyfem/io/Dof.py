# -*- coding: utf-8 -*-
"""

"""
from typing import Optional, List, Tuple

from pyfem.utils.visualization import object_slots_to_string


class Dof:
    __slots__: Tuple = ('names',
                        'family',
                        'order')

    def __init__(self) -> None:
        self.names: List[str] = None  # type: ignore
        self.family: Optional[str] = None
        self.order: Optional[int] = None

    def to_string(self, level: int = 1) -> str:
        return object_slots_to_string(self, level)

    def show(self) -> None:
        print(self.to_string())


if __name__ == "__main__":
    dof = Dof()
    dof.show()
