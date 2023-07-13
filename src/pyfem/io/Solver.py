# -*- coding: utf-8 -*-
"""

"""
from typing import Tuple

from pyfem.utils.visualization import object_slots_to_string


class Solver:
    __slots__: Tuple = ('type',
                        'option',
                        'total_time',
                        'start_time',
                        'max_increment',
                        'initial_dtime',
                        'max_dtime',
                        'min_dtime',
                        'dtype')

    def __init__(self) -> None:
        self.type: str = None  # type: ignore
        self.option: str = None  # type: ignore
        self.total_time: float = 1.0
        self.start_time: float = 0.0
        self.max_increment: int = 100
        self.initial_dtime: float = 1.0
        self.max_dtime: float = 1.0
        self.min_dtime: float = 0.001
        self.dtype: str = 'float64'

    def to_string(self, level: int = 1) -> str:
        return object_slots_to_string(self, level)

    def show(self) -> None:
        print(self.to_string())


if __name__ == "__main__":
    solver = Solver()
    solver.show()
