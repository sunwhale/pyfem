# -*- coding: utf-8 -*-
"""

"""
from numpy import empty, zeros
from numpy.linalg import norm
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
        self.dof_solution = zeros(self.assembly.total_dof_number)

    def run(self) -> None:
        self.solve()
        # self.update_field_variables()

    @show_running_time
    def solve(self) -> None:

        delta_a = zeros(self.assembly.total_dof_number)

        self.assembly.update_global_stiffness()
        self.assembly.apply_bcs()
        A = self.assembly.global_stiffness
        fext = self.assembly.fext
        fint = self.assembly.fint
        rhs = self.assembly.rhs

        da = spsolve(A, rhs - fint)
        delta_a += da

        # print(da)

        iter_dof_solution = self.dof_solution + delta_a

        self.assembly.dof_solution = iter_dof_solution
        self.assembly.update_element_data()
        self.assembly.update_fint()

        residual = norm(self.assembly.fext - self.assembly.fint)

        # print(self.assembly.fint)
        # print(residual)

    def update_field_variables(self) -> None:
        total_time = 1.0
        dtime = 0.05
        self.assembly.update_element_data()
        self.assembly.update_field_variables()


if __name__ == "__main__":
    pass
