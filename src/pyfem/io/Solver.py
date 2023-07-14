# -*- coding: utf-8 -*-
"""

"""
from typing import Tuple

from pyfem.io.BaseIO import BaseIO


class Solver(BaseIO):
    __slots__: Tuple = ('type',
                        'option',
                        'total_time',
                        'start_time',
                        'max_increment',
                        'initial_dtime',
                        'max_dtime',
                        'min_dtime')

    def __init__(self) -> None:
        super().__init__()
        self.type: str = None  # type: ignore
        self.option: str = None  # type: ignore
        self.total_time: float = None  # type: ignore
        self.start_time: float = None  # type: ignore
        self.max_increment: int = None  # type: ignore
        self.initial_dtime: float = None  # type: ignore
        self.max_dtime: float = None  # type: ignore
        self.min_dtime: float = None  # type: ignore


if __name__ == "__main__":
    solver = Solver()
    solver.show()
