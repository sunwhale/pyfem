# -*- coding: utf-8 -*-
"""

"""
from copy import deepcopy

from numpy import zeros
from numpy.linalg import norm
from scipy.sparse.linalg import splu  # type: ignore

from pyfem.assembly.Assembly import Assembly
from pyfem.fem.constants import DTYPE
from pyfem.io.Solver import Solver
from pyfem.io.write_vtk import write_vtk, write_pvd
from pyfem.solvers.BaseSolver import BaseSolver
from pyfem.utils.colors import error_style
from pyfem.utils.colors import info_style


class NonlinearSolver(BaseSolver):
    """
    非线性求解器。

    :ivar PENALTY: 罚系数
    :vartype PENALTY: float

    :ivar FORCE_TOL: 残差容限
    :vartype FORCE_TOL: float

    :ivar MAX_NITER: 最大迭代次数
    :vartype MAX_NITER: int
    """

    __slots_dict__: dict = {
        'PENALTY': ('float', '罚系数'),
        'FORCE_TOL': ('float', '残差容限'),
        'MAX_NITER': ('int', '最大迭代次数')
    }

    __slots__ = BaseSolver.__slots__ + [slot for slot in __slots_dict__.keys()]

    def __init__(self, assembly: Assembly, solver: Solver) -> None:
        super().__init__()
        self.assembly = assembly
        self.solver = solver
        self.dof_solution = zeros(self.assembly.total_dof_number)
        self.PENALTY: float = 1.0e16
        self.FORCE_TOL: float = 1.0e-6
        self.MAX_NITER: int = 32

    def run(self) -> int:
        if self.assembly.props.solver.option in [None, '', 'NR', 'NewtonRaphson']:
            return self.Newton_Raphson_solve()
        elif self.assembly.props.solver.option in ['IT', 'InitialTangent']:
            return self.initial_tangent_solve()
        else:
            raise NotImplementedError(error_style(
                f'unsupported option \'{self.assembly.props.solver.option}\' of {self.assembly.props.solver.type}'))

    def Newton_Raphson_solve(self) -> int:
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

            self.assembly.ddof_solution = zeros(self.assembly.total_dof_number, dtype=DTYPE)

            is_convergence = False
            for niter in range(self.MAX_NITER):
                self.assembly.assembly_global_stiffness()
                fint = self.assembly.fint
                rhs = deepcopy(self.assembly.fext)
                if niter == 0:
                    for bc_data in self.assembly.bc_data_list:
                        amplitude_increment = bc_data.get_amplitude(timer.time1) - bc_data.get_amplitude(timer.time0)
                        if bc_data.bc.category == 'DirichletBC':
                            for dof_id, dof_value in zip(bc_data.dof_ids, bc_data.dof_values):
                                self.assembly.global_stiffness[dof_id, dof_id] += self.PENALTY
                                rhs[dof_id] += dof_value * self.PENALTY * amplitude_increment
                        elif bc_data.bc.category == 'NeumannBC':
                            for dof_id, fext in zip(bc_data.dof_ids, bc_data.bc_fext):
                                rhs[dof_id] += fext * amplitude_increment
                                self.assembly.fext[dof_id] += fext * amplitude_increment
                else:
                    for bc_data in self.assembly.bc_data_list:
                        if bc_data.bc.category == 'DirichletBC':
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

                print(f'  niter = {niter}, residual = {f_residual}')

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
            print((error_style('maximum increment is reached')))
            return -1
        else:
            return 0

    def initial_tangent_solve(self) -> int:
        self.MAX_NITER = 1024
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
                        amplitude_increment = bc_data.get_amplitude(timer.time1) - bc_data.get_amplitude(timer.time0)
                        if bc_data.bc.category == 'DirichletBC':
                            for dof_id, dof_value in zip(bc_data.dof_ids, bc_data.dof_values):
                                self.assembly.global_stiffness[dof_id, dof_id] += self.PENALTY
                                rhs[dof_id] += dof_value * self.PENALTY * amplitude_increment
                        elif bc_data.bc.category == 'NeumannBC':
                            for dof_id, fext in zip(bc_data.dof_ids, bc_data.bc_fext):
                                rhs[dof_id] += fext * amplitude_increment
                                self.assembly.fext[dof_id] += fext * amplitude_increment
                    LU = splu(self.assembly.global_stiffness)
                else:
                    fint = self.assembly.fint
                    rhs = deepcopy(self.assembly.fext)
                    for bc_data in self.assembly.bc_data_list:
                        if bc_data.bc.category == 'DirichletBC':
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
            print((error_style('maximum increment is reached')))
            return -1
        else:
            return 0


if __name__ == "__main__":
    from pyfem.utils.visualization import print_slots_dict

    print_slots_dict(NonlinearSolver.__slots_dict__)

    from pyfem.Job import Job

    job = Job(r'..\..\..\examples\mechanical\plane\Job-1.toml')
    solver = NonlinearSolver(job.assembly, job.props.solver)
    solver.show()
