# -*- coding: utf-8 -*-
"""

"""
from numpy import empty
from scipy.sparse.linalg import spsolve  # type: ignore

from pyfem.assembly.Assembly import Assembly
from pyfem.io.Solver import Solver
from pyfem.solvers.BaseSolver import BaseSolver
from pyfem.utils.wrappers import show_running_time


class NonlinearSolver(BaseSolver):
    def __init__(self, assembly: Assembly, solver: Solver) -> None:
        super().__init__(assembly, solver)
        self.assembly: Assembly = assembly
        self.solver: Solver = solver
        self.solution = empty(0)

    def run(self) -> None:
        self.solve()
        self.update_field_variables()

    @show_running_time
    def solve(self) -> None:
        A = self.assembly.global_stiffness
        rhs = self.assembly.fext
        total_time = 1.0
        dtime = 0.05
        x = spsolve(A, rhs)
        self.solution = x
        self.assembly.dof_solution = x

    def update_field_variables(self) -> None:
        self.assembly.update_element_data()
        self.assembly.update_field_variables()


if __name__ == "__main__":
    pass