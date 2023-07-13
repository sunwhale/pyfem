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
    __slots__ = BaseSolver.__slots__ + ('PENALTY',)

    def __init__(self, assembly: Assembly, solver: Solver) -> None:
        super().__init__()
        self.assembly = assembly
        self.solver = solver
        self.dof_solution = empty(0, dtype=DTYPE)
        self.PENALTY: float = 1.0e16

    def run(self) -> int:
        return self.solve()

    def solve(self) -> int:
        A = self.assembly.global_stiffness
        rhs = self.assembly.fext

        for bc_data in self.assembly.bc_data_list:
            if bc_data.bc.category == 'DirichletBC':
                for dof_id, dof_value in zip(bc_data.dof_ids, bc_data.dof_values):
                    A[dof_id, dof_id] += self.PENALTY
                    rhs[dof_id] += dof_value * self.PENALTY
            elif bc_data.bc.category == 'NeumannBC':
                for dof_id, fext in zip(bc_data.dof_ids, bc_data.bc_fext):
                    rhs[dof_id] += fext

        x = spsolve(A, rhs)

        self.dof_solution = x
        self.assembly.dof_solution = x
        self.assembly.update_element_data()
        self.assembly.update_element_field_variables()
        self.assembly.assembly_field_variables()
        write_vtk(self.assembly)

        return 0


if __name__ == "__main__":
    from pyfem.Job import Job

    job = Job(r'..\..\..\examples\mechanical\plane\Job-1.toml')
    solver = LinearSolver(job.assembly, job.props.solver)
    solver.show()
