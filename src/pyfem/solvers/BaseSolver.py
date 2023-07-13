# -*- coding: utf-8 -*-
"""

"""
from typing import Tuple

from numpy import ndarray, empty

from pyfem.assembly.Assembly import Assembly
from pyfem.io.Solver import Solver
from pyfem.utils.visualization import object_slots_to_string_ndarray


class BaseSolver:
    __slots__: Tuple = ('assembly',
                        'solver',
                        'dof_solution')

    def __init__(self) -> None:
        self.assembly: Assembly = None  # type: ignore
        self.solver: Solver = None  # type: ignore
        self.dof_solution: ndarray = empty(0)

    def to_string(self, level: int = 1) -> str:
        return object_slots_to_string_ndarray(self, level)

    def show(self) -> None:
        print(self.to_string())

    def run(self) -> int:
        return -1

    def solve(self) -> int:
        return -1


if __name__ == "__main__":
    solver = BaseSolver()
    solver.show()
