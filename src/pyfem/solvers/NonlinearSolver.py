# -*- coding: utf-8 -*-
"""

"""
from copy import deepcopy

from numpy import zeros
from numpy.linalg import norm
from scipy.sparse.linalg import splu  # type: ignore

from pyfem.assembly.Assembly import Assembly
from pyfem.io.Solver import Solver
from pyfem.io.write_vtk import write_vtk, write_pvd
from pyfem.solvers.BaseSolver import BaseSolver
from pyfem.utils.colors import error_style
from pyfem.utils.colors import info_style
from pyfem.utils.wrappers import show_running_time


class NonlinearSolver(BaseSolver):
    def __init__(self, assembly: Assembly, solver: Solver) -> None:
        super().__init__(assembly, solver)
        self.assembly: Assembly = assembly
        self.solver: Solver = solver
        self.dof_solution = zeros(self.assembly.total_dof_number)
        self.PENALTY = 1.0e16
        self.FORCE_TOL = 1.0e-6
        self.MAX_NITER = 64

    def run(self) -> None:
        self.Newton_Raphson_solve()

    @show_running_time
    def initial_tangent_solve(self) -> None:
        timer = self.assembly.timer

        timer.total_time = self.solver.total_time
        timer.dtime = self.solver.initial_dtime
        timer.time0 = self.solver.start_time
        timer.increment = 0
        timer.frame_ids.append(0)

        self.assembly.update_element_field_variables()
        self.assembly.assembly_field_variables()
        write_vtk(self.assembly)

        for increment in range(1, self.solver.max_increment):

            timer.time1 = timer.time0 + timer.dtime
            timer.increment = increment

            print(info_style(f'increment = {increment}, time = {timer.time1}'))

            self.assembly.ddof_solution = zeros(self.assembly.total_dof_number)

            is_convergence = False
            for niter in range(self.MAX_NITER):
                if niter == 0:
                    self.assembly.assembly_global_stiffness()
                    fint = self.assembly.fint
                    rhs = deepcopy(self.assembly.fext)
                    for bc_data in self.assembly.bc_data_list:
                        for dof_id, dof_value in zip(bc_data.dof_ids, bc_data.dof_values):
                            self.assembly.global_stiffness[dof_id, dof_id] += self.PENALTY
                            rhs[dof_id] += dof_value * timer.dtime / timer.total_time * self.PENALTY
                    LU = splu(self.assembly.global_stiffness)
                else:
                    fint = self.assembly.fint
                    rhs = deepcopy(self.assembly.fext)
                    for bc_data in self.assembly.bc_data_list:
                        for dof_id, dof_value in zip(bc_data.dof_ids, bc_data.dof_values):
                            rhs[dof_id] = 0.0 * self.PENALTY

                da = LU.solve(rhs - fint)

                self.assembly.ddof_solution += da
                self.assembly.update_element_data_without_stiffness()
                self.assembly.assembly_fint()

                f_residual = self.assembly.fext - self.assembly.fint
                f_residual[self.assembly.bc_dof_ids] = 0
                f_residual = norm(f_residual)

                print(f'  niter = {niter}, force residual = {f_residual}')

                if f_residual < self.FORCE_TOL:
                    is_convergence = True
                    break

            if is_convergence:
                self.assembly.dof_solution += self.assembly.ddof_solution
                self.assembly.update_element_data()
                self.assembly.update_element_state_variables()
                self.assembly.update_element_field_variables()
                self.assembly.assembly_field_variables()
            else:
                raise NotImplementedError(error_style('the iteration is not convergence'))

            write_vtk(self.assembly)

            timer.time0 = timer.time1
            timer.frame_ids.append(increment)

            if timer.is_done():
                write_pvd(self.assembly)
                break

        if not timer.is_done():
            raise NotImplementedError(error_style('maximum increment is reached'))

    def Newton_Raphson_solve(self) -> None:
        timer = self.assembly.timer
        timer.total_time = self.solver.total_time
        timer.dtime = self.solver.initial_dtime
        timer.time0 = self.solver.start_time
        timer.increment = 0
        timer.frame_ids.append(0)

        self.assembly.update_element_field_variables()
        self.assembly.assembly_field_variables()
        write_vtk(self.assembly)

        for increment in range(1, self.solver.max_increment):

            timer.time1 = timer.time0 + timer.dtime
            timer.increment = increment

            print(info_style(f'increment = {increment}, time = {timer.time1}'))

            self.assembly.ddof_solution = zeros(self.assembly.total_dof_number)

            is_convergence = False
            for niter in range(self.MAX_NITER):
                self.assembly.assembly_global_stiffness()
                fint = self.assembly.fint
                rhs = deepcopy(self.assembly.fext)
                if niter == 0:
                    for bc_data in self.assembly.bc_data_list:
                        for dof_id, dof_value in zip(bc_data.dof_ids, bc_data.dof_values):
                            self.assembly.global_stiffness[dof_id, dof_id] += self.PENALTY
                            rhs[dof_id] += dof_value * timer.dtime / timer.total_time * self.PENALTY
                else:
                    for bc_data in self.assembly.bc_data_list:
                        for dof_id, dof_value in zip(bc_data.dof_ids, bc_data.dof_values):
                            self.assembly.global_stiffness[dof_id, dof_id] += self.PENALTY
                            rhs[dof_id] = 0.0 * self.PENALTY

                LU = splu(self.assembly.global_stiffness)
                da = LU.solve(rhs - fint)

                self.assembly.ddof_solution += da
                self.assembly.update_element_data()
                self.assembly.assembly_fint()

                f_residual = self.assembly.fext - self.assembly.fint
                f_residual[self.assembly.bc_dof_ids] = 0
                f_residual = norm(f_residual)

                print(f'  niter = {niter}, force residual = {f_residual}')

                if f_residual < self.FORCE_TOL:
                    is_convergence = True
                    break

            if is_convergence:
                self.assembly.dof_solution += self.assembly.ddof_solution
                self.assembly.update_element_data()
                self.assembly.update_element_state_variables()
                self.assembly.update_element_field_variables()
                self.assembly.assembly_field_variables()
            else:
                raise NotImplementedError(error_style('the iteration is not convergence'))

            write_vtk(self.assembly)

            timer.time0 = timer.time1
            timer.frame_ids.append(increment)

            if timer.is_done():
                write_pvd(self.assembly)
                break

        if not timer.is_done():
            raise NotImplementedError(error_style('maximum increment is reached'))


if __name__ == "__main__":
    pass
