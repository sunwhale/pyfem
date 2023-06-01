# -*- coding: utf-8 -*-
"""

"""
from numpy import empty
from scipy.sparse.linalg import spsolve  # type: ignore

from pyfem.assembly.Assembly import Assembly
from pyfem.io.Solver import Solver
from pyfem.solvers.BaseSolver import BaseSolver
from pyfem.utils.wrappers import show_running_time
from pyfem.io.write_vtk import write_vtk, write_pvd


class LinearSolver(BaseSolver):
    def __init__(self, assembly: Assembly, solver: Solver) -> None:
        super().__init__(assembly, solver)
        self.assembly: Assembly = assembly
        self.solver: Solver = solver
        self.dof_solution = empty(0)

    def run(self) -> None:
        self.solve()

    @show_running_time
    def solve(self) -> None:
        self.assembly.apply_bcs()
        A = self.assembly.global_stiffness
        rhs = self.assembly.rhs
        x = spsolve(A, rhs)
        self.dof_solution = x
        self.assembly.dof_solution = x
        self.assembly.update_element_data()
        self.assembly.update_field_variables()
        write_vtk(self.assembly)


if __name__ == "__main__":
    pass
