# -*- coding: utf-8 -*-
"""

"""
from copy import deepcopy

from numpy import zeros
from numpy.linalg import norm
from scipy.sparse.linalg import spsolve  # type: ignore

from pyfem.assembly.Assembly import Assembly
from pyfem.fem.Timer import Timer
from pyfem.io.Solver import Solver
from pyfem.solvers.BaseSolver import BaseSolver
from pyfem.utils.colors import info_style
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
        PENALTY = 1.0e16
        MAX_NITER = 25
        MAX_NINC = 100
        F_TOL = 1.0e-6

        total_time = 1.0
        start_time = 0.0
        dtime = 0.1

        timer = self.assembly.timer
        timer.total_time = total_time
        timer.dtime = dtime
        timer.time0 = start_time

        for ninc in range(MAX_NINC):

            timer.time1 = timer.time0 + timer.dtime

            delta_a = zeros(self.assembly.total_dof_number)

            print(info_style(f'ninc = {ninc}, time = {timer.time1}'))

            for niter in range(MAX_NITER):
                self.assembly.update_global_stiffness()
                fint = self.assembly.fint
                rhs = deepcopy(self.assembly.fext)

                for bc_data in self.assembly.bc_data_list:
                    for dof_id, dof_value in zip(bc_data.dof_ids, bc_data.dof_values):
                        self.assembly.global_stiffness[dof_id, dof_id] += PENALTY
                        if niter == 0:
                            rhs[dof_id] += dof_value * timer.dtime / total_time * PENALTY
                        else:
                            rhs[dof_id] = 0.0 * PENALTY

                A = self.assembly.global_stiffness

                da = spsolve(A, rhs - fint)

                delta_a += da

                self.assembly.ddof_solution = delta_a

                self.assembly.update_element_data()

                self.assembly.update_fint()

                f_residual = self.assembly.fext - self.assembly.fint
                f_residual[self.assembly.bc_dof_ids] = 0
                f_residual = norm(f_residual)

                print(f'  niter = {niter}, residual force = {f_residual}')

                if f_residual < F_TOL:
                    break

            self.assembly.dof_solution += delta_a
            self.assembly.update_element_data()
            self.assembly.update_state_variables()

            timer.show()

            timer.time0 = timer.time1

            if timer.is_done():
                break

    def update_field_variables(self) -> None:
        total_time = 1.0
        dtime = 0.05
        self.assembly.update_element_data()
        self.assembly.update_field_variables()


if __name__ == "__main__":
    pass
