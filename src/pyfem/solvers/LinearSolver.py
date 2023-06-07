# -*- coding: utf-8 -*-
"""

"""
from numpy import empty
from scipy.sparse.linalg import spsolve  # type: ignore

from pyfem.assembly.Assembly import Assembly
from pyfem.fem.constants import DTYPE
from pyfem.io.Solver import Solver
from pyfem.io.write_vtk import write_vtk
from pyfem.solvers.BaseSolver import BaseSolver


class LinearSolver(BaseSolver):
    def __init__(self, assembly: Assembly, solver: Solver) -> None:
        super().__init__(assembly, solver)
        self.assembly: Assembly = assembly
        self.solver: Solver = solver
        self.dof_solution = empty(0, dtype=DTYPE)
        self.PENALTY = 1.0e16

    def run(self) -> None:
        self.solve()

    def solve(self) -> None:
        A = self.assembly.global_stiffness
        rhs = self.assembly.fext

        for bc_data in self.assembly.bc_data_list:
            for dof_id, dof_value in zip(bc_data.dof_ids, bc_data.dof_values):
                A[dof_id, dof_id] += self.PENALTY
                rhs[dof_id] += dof_value * self.PENALTY

        x = spsolve(A, rhs)
        self.dof_solution = x
        self.assembly.dof_solution = x
        self.assembly.update_element_data_without_stiffness()
        self.assembly.update_element_field_variables()
        self.assembly.assembly_field_variables()
        write_vtk(self.assembly)


if __name__ == "__main__":
    pass
