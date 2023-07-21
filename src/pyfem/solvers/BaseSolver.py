# -*- coding: utf-8 -*-
"""

"""
from numpy import ndarray, empty

from pyfem.assembly.Assembly import Assembly
from pyfem.io.Solver import Solver
from pyfem.utils.visualization import object_slots_to_string_ndarray


class BaseSolver:
    """
    求解器基类。

    :ivar assembly: 装配体对象
    :vartype assembly: Assembly

    :ivar solver: 求解器属性
    :vartype solver: Solver

    :ivar dof_solution: 求解得到自由度的值
    :vartype dof_solution: ndarray
    """

    __slots_dict__: dict = {
        'assembly': ('Assembly', '装配体对象'),
        'solver': ('Solver', '求解器属性'),
        'dof_solution': ('ndarray', '求解得到自由度的值')
    }

    __slots__: list = [slot for slot in __slots_dict__.keys()]

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
    from pyfem.utils.visualization import print_slots_dict

    print_slots_dict(BaseSolver.__slots_dict__)

    solver = BaseSolver()
    solver.show()
