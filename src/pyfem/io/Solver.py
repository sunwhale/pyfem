# -*- coding: utf-8 -*-
"""

"""
from pyfem.utils.visualization import object_dict_to_string


class Solver:
    def __init__(self) -> None:
        self.type: str = None  # type: ignore
        self.option: str = None  # type: ignore
        self.total_time: float = 1.0
        self.start_time: float = 0.0
        self.max_increment: int = 100
        self.initial_dtime: float = 1.0
        self.max_dtime: float = 1.0
        self.min_dtime: float = 0.001

    def to_string(self, level: int = 1) -> str:
        return object_dict_to_string(self, level)

    def show(self) -> None:
        print(self.to_string())


if __name__ == "__main__":
    solver = Solver()
    print(solver.__dict__.keys())
    print(solver)
    print(solver.to_string())
