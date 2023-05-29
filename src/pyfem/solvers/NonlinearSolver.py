# -*- coding: utf-8 -*-
"""

"""
from numpy import empty, zeros, array
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
        self.update_field_variables()

    @show_running_time
    def solve(self) -> None:
        penalty = 1.0e16
        delta_a = zeros(self.assembly.total_dof_number)

        for i in range(25):
            self.assembly.update_global_stiffness()
            fint = self.assembly.fint
            rhs = deepcopy(self.assembly.fext)

            for bc_data in self.assembly.bc_data_list:
                for dof_id, dof_value in zip(bc_data.dof_ids, bc_data.dof_values):
                    self.assembly.global_stiffness[dof_id, dof_id] += penalty
                    if i == 0:
                        rhs[dof_id] += dof_value * penalty
                    else:
                        rhs[dof_id] = 0.0 * penalty

            A = self.assembly.global_stiffness

            da = spsolve(A, rhs - fint)

            delta_a += da

            self.assembly.ddof_solution = delta_a

            self.assembly.update_element_data()

            self.assembly.update_fint()

            residual = self.assembly.fext - self.assembly.fint
            residual[self.assembly.bc_dof_ids] = 0
            residual = norm(residual)

            print(f'iteration = {i}, residual = {residual}')

            if residual < 1.0e-6:
                break

    # def solve(self) -> None:
    #     for i in range(25):
    #         print(i)
    #         self.assembly.update_global_stiffness()
    #         A = self.assembly.global_stiffness
    #         fext = self.assembly.fext
    #         fint = self.assembly.fint
    #         rhs = self.assembly.rhs
    #         penalty = 1.0e32
    #         bc_dof_ids = []
    #         if i == 0:
    #             for bc_data in self.assembly.bc_data_list:
    #                 for dof_id, dof_value in zip(bc_data.dof_ids, bc_data.dof_values):
    #                     bc_dof_ids.append(dof_id)
    #                     self.assembly.global_stiffness[dof_id, dof_id] += penalty
    #                     rhs[dof_id] += dof_value * penalty
    #             self.assembly.bc_dof_ids = array(bc_dof_ids)
    #         else:
    #             for bc_data in self.assembly.bc_data_list:
    #                 for dof_id, dof_value in zip(bc_data.dof_ids, bc_data.dof_values):
    #                     self.assembly.global_stiffness[dof_id, dof_id] += penalty
    #                     rhs[dof_id] = 0.0 * penalty
    #
    #         da = spsolve(A, rhs - fint)
    #
    #         self.assembly.dof_solution += da
    #         self.assembly.update_element_data()
    #         self.assembly.update_fint()
    #
    #         print(f'dof_solution = {self.assembly.dof_solution}')
    #
    #         fint = deepcopy(self.assembly.fint)
    #         fint[self.assembly.bc_dof_ids] = 0
    #         fext = deepcopy(self.assembly.fext)
    #         residual = norm(fext - fint)
    #
    #         print(f'residual = {residual}')

    def update_field_variables(self) -> None:
        total_time = 1.0
        dtime = 0.05
        self.assembly.update_element_data()
        self.assembly.update_field_variables()


if __name__ == "__main__":
    pass
