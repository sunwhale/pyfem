# -*- coding: utf-8 -*-
"""

"""
from typing import List, Tuple

from pyfem.utils.visualization import object_slots_to_string


class Amplitude:
    __slots__: Tuple = ('name',
                        'type',
                        'start',
                        'data')

    def __init__(self) -> None:
        self.name: str = None  # type: ignore
        self.type: str = None  # type: ignore
        self.start: float = 0.0
        self.data: List = None  # type: ignore

    def to_string(self, level: int = 1) -> str:
        return object_slots_to_string(self, level)

    def show(self) -> None:
        print(self.to_string())


if __name__ == "__main__":
    amp = Amplitude()
    amp.show()
