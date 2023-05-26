# -*- coding: utf-8 -*-
"""

"""
from numpy import empty, zeros
from copy import deepcopy
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

        for i in range(4):
            self.assembly.update_global_stiffness()
            self.assembly.apply_bcs()
            A = self.assembly.global_stiffness
            fext = self.assembly.fext
            fint = self.assembly.fint
            rhs = self.assembly.rhs

            da = spsolve(A, rhs - fint)

            # delta_a += da

            print(f'fint = {fint}')

            # iter_dof_solution = self.dof_solution + delta_a

            self.assembly.dof_solution += da
            self.assembly.update_element_data()
            self.assembly.update_fint()

            fint = deepcopy(self.assembly.fint)
            fint[self.assembly.bc_dof_ids] = 0
            residual = norm(fext - fint)

            # print('rhs=', rhs)
            # print(self.assembly.fext)
            print(fint)

            print(residual)

    # def solve(self) -> None:
    #     from copy import deepcopy
    #     delta_a = zeros(self.assembly.total_dof_number)
    #
    #     for i in range(10):
    #         self.assembly.update_global_stiffness()
    #         A0 = deepcopy(self.assembly.global_stiffness)
    #         self.assembly.apply_bcs()
    #         A = self.assembly.global_stiffness
    #         fext = self.assembly.fext
    #         fint = self.assembly.fint
    #         rhs = self.assembly.rhs
    #
    #         da = spsolve(A, rhs - fint)
    #         delta_a += da
    #
    #         iter_dof_solution = self.dof_solution + delta_a
    #
    #         self.assembly.dof_solution = iter_dof_solution
    #         self.assembly.update_element_data()
    #         self.assembly.update_fint()
    #
    #         # print(self.assembly.bc_dof_ids)
    #         self.assembly.fint[self.assembly.bc_dof_ids] = 0
    #         residual = norm(self.assembly.fext - self.assembly.fint)
    #
    #         print(self.assembly.fint)
    #
    #         # x = A0.dot(da)
    #         # x[self.assembly.bc_dof_ids] = 0
    #         # print(x)
    #         # print(A.dot(da))
    #         print(residual)
    def update_field_variables(self) -> None:
        total_time = 1.0
        dtime = 0.05
        self.assembly.update_element_data()
        self.assembly.update_field_variables()


if __name__ == "__main__":
    pass
