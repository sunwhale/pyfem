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

        for i in range(50):
            print(i)
            self.assembly.update_global_stiffness()
            self.assembly.fext[[5, 7]] = 150.0
            self.assembly.fext[[13]] = 300.0
            fext = self.assembly.fext
            fint = self.assembly.fint
            rhs = deepcopy(fext)

            bc_dof_ids = []
            if i == 0:
                for bc_data in self.assembly.bc_data_list:
                    for dof_id, dof_value in zip(bc_data.dof_ids, bc_data.dof_values):
                        bc_dof_ids.append(dof_id)
                        self.assembly.global_stiffness[dof_id, dof_id] += penalty
                        rhs[dof_id] += dof_value * penalty
                self.assembly.bc_dof_ids = array(bc_dof_ids)
            else:
                for bc_data in self.assembly.bc_data_list:
                    for dof_id, dof_value in zip(bc_data.dof_ids, bc_data.dof_values):
                        self.assembly.global_stiffness[dof_id, dof_id] += penalty
                        rhs[dof_id] += 0.0 * penalty

            A = self.assembly.global_stiffness

            da = spsolve(A, rhs - fint)
            delta_a += da

            self.assembly.ddof_solution = delta_a
            # self.assembly.dof_solution += da
            self.assembly.update_element_data()
            self.assembly.update_fint()

            fint = deepcopy(self.assembly.fint)
            fint[self.assembly.bc_dof_ids] = 0
            fext = deepcopy(self.assembly.fext)

            fint[[5, 7, 13]] = 0
            fext[[5, 7, 13]] = 0

            residual = norm(fext - fint)

            print(f'residual = {residual}')

            if residual < 1.0e-6:
                break

    # def solve(self) -> None:
    #
    #     for i in range(5):
    #         print(i)
    #         self.assembly.update_global_stiffness()
    #         # self.assembly.fext[[5, 7]] = 110.0
    #         # self.assembly.fext[[13]] = 220.0
    #         # self.assembly.apply_bcs()
    #
    #         fext = self.assembly.fext
    #         fint = self.assembly.fint
    #         # rhs = self.assembly.rhs
    #
    #         bc_dof_ids = []
    #         penalty = 1.0e32
    #         rhs = deepcopy(fext)
    #
    #         if i == 0:
    #             for bc_data in self.assembly.bc_data_list:
    #                 for dof_id, dof_value in zip(bc_data.dof_ids, bc_data.dof_values):
    #                     bc_dof_ids.append(dof_id)
    #                     self.assembly.global_stiffness[dof_id, dof_id] += penalty
    #                     rhs[dof_id] += dof_value * penalty
    #                     # print(dof_id, dof_value)
    #             self.assembly.bc_dof_ids = array(bc_dof_ids)
    #         else:
    #             for bc_data in self.assembly.bc_data_list:
    #                 for dof_id, dof_value in zip(bc_data.dof_ids, bc_data.dof_values):
    #                     self.assembly.global_stiffness[dof_id, dof_id] += penalty
    #                     rhs[dof_id] += 0.0 * penalty
    #
    #         A = self.assembly.global_stiffness
    #
    #         da = spsolve(A, rhs - fint)
    #
    #         self.assembly.dof_solution += da
    #
    #         # print(f'dof_solution = {self.assembly.dof_solution}')
    #         # print(f'fext = {fext}')
    #         # print(f'fint = {fint}')
    #
    #         self.assembly.update_element_data()
    #         self.assembly.update_fint()
    #
    #         fint = deepcopy(self.assembly.fint)
    #         fint[self.assembly.bc_dof_ids] = 0
    #
    #         # print(f'fint = {fint}')
    #         # print(f'fint = {self.assembly.fint}')
    #         fext = deepcopy(self.assembly.fext)
    #         # fint[[5, 7, 13]] = 0
    #         # fext[[5, 7, 13]] = 0
    #         residual = norm(fext - fint)
    #
    #         # print('rhs=', rhs)
    #         # print(self.assembly.fext)
    #
    #         print(f'residual = {residual}')

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
