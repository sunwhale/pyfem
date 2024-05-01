# -*- coding: utf-8 -*-
"""

"""
import time
from copy import deepcopy

from numpy import zeros
from numpy.linalg import norm
from scipy.sparse.linalg import splu  # type: ignore

from pyfem.assembly.Assembly import Assembly
from pyfem.fem.constants import DTYPE, IS_PETSC
from pyfem.io.Solver import Solver
from pyfem.io.write_vtk import write_vtk, write_pvd
from pyfem.solvers.BaseSolver import BaseSolver
from pyfem.utils.colors import error_style
from pyfem.utils.logger import logger, logger_sta
from pyfem.io.write_hdf5 import add_hdf5
from pyfem.database.Database import Database


class NonlinearSolver(BaseSolver):
    r"""
    非线性求解器。

    :ivar PENALTY: 罚系数
    :vartype PENALTY: float

    :ivar FORCE_TOL: 残差容限
    :vartype FORCE_TOL: float

    :ivar MAX_NITER: 最大迭代次数
    :vartype MAX_NITER: int

    :ivar BC_METHOD: 边界条件施加方式
    :vartype BC_METHOD: str

    对于准静态过程，需要求解的线性系统可以表示为：

    .. math::
        {{\mathbf{f}}_{{\text{ext}}}} - {{\mathbf{f}}_{{\text{int}}}} = {\mathbf{0}}

    虽然在静态力学过程中，时间不再起作用，但是我们仍然需要一个参数来排列事件的顺序。出于这个原因，我们将继续在静态力学过程中使用“时间”的概念来表示加载顺序。特别地，时间的概念可以用来将完整的外部载荷分解为若干个增量加载步骤。这样做的原因如下：

    1. 从非线性连续体模型的离散化得到的代数方程组是非线性的，因此需要使用迭代过程来求解。对于非常大的加载步骤（例如在一个增量步中施加整个载荷），通常很难获得一个正确收敛的解。对于大多数常用的迭代过程，包括牛顿-拉夫逊方法，其收敛半径都是有限的。

    2. 实验证明，大多数材料表现出路径相关的行为。这意味着根据所遵循的应变路径获得的应力值是不同的。例如，当我们首先对一个平板施加拉伸应变增量，然后施加剪切应变增量时，所得到的应力可能会不同，或者当以相反的顺序施加相同的应变增量时，结果应力也可能不同。显然，只有在应变增量相对较小的情况下，才能正确预测结构行为，以便尽可能地按照应变路径进行。

    参考文献：

    [1] Non‐Linear Finite Element Analysis of Solids and Structures, John Wiley & Sons, Ltd, 2012, 31-62, https://doi.org/10.1002/9781118375938.ch2
    """

    __slots_dict__: dict = {
        'PENALTY': ('float', '罚系数'),
        'FORCE_TOL': ('float', '残差容限'),
        'MAX_NITER': ('int', '最大迭代次数'),
        'BC_METHOD': ('str', '边界条件施加方式')
    }

    __slots__ = BaseSolver.__slots__ + [slot for slot in __slots_dict__.keys()]

    def __init__(self, assembly: Assembly, solver: Solver) -> None:
        super().__init__()
        self.assembly = assembly
        self.solver = solver
        self.dof_solution = zeros(self.assembly.total_dof_number)
        self.database = Database(self.assembly)
        self.PENALTY: float = 1.0e128
        self.FORCE_TOL: float = 1.0e-3
        self.MAX_NITER: int = 16
        self.BC_METHOD: str = '01'

    def run(self) -> int:
        if self.assembly.props.solver.option in [None, '', 'NR', 'NewtonRaphson']:
            return self.incremental_iterative_solve('NR')
        elif self.assembly.props.solver.option in ['IT', 'InitialTangent']:
            return self.incremental_iterative_solve('IT')
        else:
            raise NotImplementedError(error_style(
                f'unsupported option \'{self.assembly.props.solver.option}\' of {self.assembly.props.solver.type}'))

    def incremental_iterative_solve(self, option: str) -> int:
        timer = self.assembly.timer
        timer.total_time = self.solver.total_time
        timer.dtime = self.solver.initial_dtime
        timer.time0 = self.solver.start_time
        timer.increment = 0
        timer.frame_ids.append(0)

        self.assembly.update_element_field_variables()
        self.assembly.assembly_field_variables()
        self.database.init_hdf5()
        for output in self.assembly.props.outputs:
            if output.is_save:
                if output.type == 'vtk':
                    write_vtk(self.assembly)

        increment: int = 1
        attempt: int = 1

        for i in range(1, self.solver.max_increment):

            timer.time1 = timer.time0 + timer.dtime
            timer.increment = increment

            logger.info(f'increment = {increment}, attempt = {attempt}, time = {timer.time1:14.9f}, dtime = {timer.dtime:14.9f}')

            self.assembly.ddof_solution = zeros(self.assembly.total_dof_number, dtype=DTYPE)
            self.assembly.update_element_data()

            is_convergence = False

            for niter in range(self.MAX_NITER):
                self.assembly.assembly_global_stiffness()
                fint = deepcopy(self.assembly.fint)
                rhs = deepcopy(self.assembly.fext)

                # 罚系数法施加边界条件
                if self.BC_METHOD == 'PENALTY':
                    if niter == 0:
                        for bc_data in self.assembly.bc_data_list:
                            amplitude_increment = bc_data.get_amplitude(timer.time1) - bc_data.get_amplitude(timer.time0)
                            if bc_data.bc.category == 'DirichletBC':
                                for bc_dof_id, bc_dof_value in zip(bc_data.bc_dof_ids, bc_data.bc_dof_values):
                                    self.assembly.global_stiffness[bc_dof_id, bc_dof_id] += self.PENALTY
                                    rhs[bc_dof_id] += bc_dof_value * self.PENALTY * amplitude_increment
                            elif bc_data.bc.category == 'NeumannBC':
                                for bc_dof_id, bc_fext in zip(bc_data.bc_dof_ids, bc_data.bc_fext):
                                    rhs[bc_dof_id] += bc_fext * amplitude_increment
                                    self.assembly.fext[bc_dof_id] += bc_fext * amplitude_increment
                    else:
                        for bc_data in self.assembly.bc_data_list:
                            if bc_data.bc.category == 'DirichletBC':
                                for bc_dof_id, bc_dof_value in zip(bc_data.bc_dof_ids, bc_data.bc_dof_values):
                                    self.assembly.global_stiffness[bc_dof_id, bc_dof_id] += self.PENALTY
                                    rhs[bc_dof_id] = 0.0 * self.PENALTY

                # 划0置1法
                if self.BC_METHOD == '01':
                    if IS_PETSC:
                        b = self.assembly.A.createVecLeft()
                        x = self.assembly.A.createVecRight()
                        b.setValues(range(self.assembly.total_dof_number), self.assembly.fext)
                    else:
                        self.assembly.global_stiffness = self.assembly.global_stiffness.tolil()

                    if niter == 0:
                        for bc_data in self.assembly.bc_data_list:
                            amplitude_increment = bc_data.get_amplitude(timer.time1) - bc_data.get_amplitude(timer.time0)
                            if bc_data.bc.category == 'DirichletBC':
                                if IS_PETSC:
                                    x.setValues(bc_data.bc_dof_ids, bc_data.bc_dof_values * amplitude_increment)
                                    self.assembly.A.zeroRowsColumns(bc_data.bc_dof_ids, diag=1.0, x=x, b=b)
                                    b.setValues(bc_data.bc_dof_ids, bc_data.bc_dof_values * amplitude_increment + fint[bc_data.bc_dof_ids])
                                else:
                                    for bc_dof_id, bc_dof_value in zip(bc_data.bc_dof_ids, bc_data.bc_dof_values):
                                        rhs -= self.assembly.global_stiffness[:, bc_dof_id].toarray().reshape(-1) * bc_dof_value * amplitude_increment
                                        self.assembly.global_stiffness[bc_dof_id, :] = 0.0
                                        self.assembly.global_stiffness[:, bc_dof_id] = 0.0
                                        self.assembly.global_stiffness[bc_dof_id, bc_dof_id] = 1.0
                                        rhs[bc_dof_id] = bc_dof_value * amplitude_increment + fint[bc_dof_id]
                                    # rhs -= sum((self.assembly.A.getValues(range(self.assembly.total_dof_number), bc_data.bc_dof_ids) * bc_data.bc_dof_values * amplitude_increment), axis=1)
                                    # rhs[bc_data.bc_dof_ids] = bc_data.bc_dof_values * amplitude_increment + fint[bc_data.bc_dof_ids]
                            elif bc_data.bc.category == 'NeumannBC':
                                if IS_PETSC:
                                    b.setValues(bc_data.bc_dof_ids, bc_data.bc_fext * amplitude_increment, addv=True)
                                    self.assembly.fext[bc_data.bc_dof_ids] += bc_data.bc_fext * amplitude_increment
                                else:
                                    for bc_dof_id, bc_fext in zip(bc_data.bc_dof_ids, bc_data.bc_fext):
                                        rhs[bc_dof_id] += bc_fext * amplitude_increment
                                        self.assembly.fext[bc_dof_id] += bc_fext * amplitude_increment

                    else:
                        for bc_data in self.assembly.bc_data_list:
                            if bc_data.bc.category == 'DirichletBC':
                                if IS_PETSC:
                                    b.setValues(bc_data.bc_dof_ids, fint[bc_data.bc_dof_ids])
                                    self.assembly.A.zeroRowsColumns(bc_data.bc_dof_ids)
                                else:
                                    for bc_dof_id, bc_dof_value in zip(bc_data.bc_dof_ids, bc_data.bc_dof_values):
                                        self.assembly.global_stiffness[bc_dof_id, :] = 0.0
                                        self.assembly.global_stiffness[:, bc_dof_id] = 0.0
                                        self.assembly.global_stiffness[bc_dof_id, bc_dof_id] = 1.0
                                        rhs[bc_dof_id] = fint[bc_dof_id]
                                    # rhs[bc_data.bc_dof_ids] = fint[bc_data.bc_dof_ids]

                try:
                    if IS_PETSC:
                        try:
                            from petsc4py import PETSc  # type: ignore
                        except:
                            raise ImportError(error_style('petsc4py can not be imported'))
                        self.assembly.A.assemble()
                        ksp = PETSc.KSP().create()
                        ksp.setOperators(self.assembly.A)
                        b.setValues(range(self.assembly.total_dof_number), -fint, addv=True)
                        # ksp.setType('preonly')
                        # ksp.setConvergenceHistory()
                        # ksp.getPC().setType('lu')
                        ksp.setType('bcgs')
                        ksp.setConvergenceHistory()
                        ksp.getPC().setType('sor')
                        ksp.solve(b, x)
                        da = x.array[:]
                    else:
                        self.assembly.global_stiffness = self.assembly.global_stiffness.tocsc()
                        LU = splu(self.assembly.global_stiffness)
                        da = LU.solve(rhs - fint)

                except RuntimeError as e:
                    is_convergence = False
                    print(error_style(f"Catch RuntimeError exception: {e}"))
                    break

                self.assembly.ddof_solution += da
                if option == 'NR':
                    self.assembly.update_element_data()
                elif option == 'IT':
                    self.assembly.update_element_data_without_stiffness()
                self.assembly.assembly_fint()

                f_residual = self.assembly.fext - self.assembly.fint
                f_residual[self.assembly.bc_dof_ids] = 0
                if norm(self.assembly.fext) < 1.0e-16:
                    f_residual = norm(f_residual)
                else:
                    f_residual = norm(f_residual) / norm(self.assembly.fext)
                # f_residual = max(abs(f_residual))

                logger.log(21, f'  niter = {niter}, residual = {f_residual}')

                if timer.is_reduce_dtime:
                    timer.is_reduce_dtime = False
                    is_convergence = False
                    break

                if f_residual < self.FORCE_TOL:
                    is_convergence = True
                    break

            if is_convergence:
                logger.info(f'  increment {increment} is convergence')
                logger_sta.info(f'{1:4}  {increment:9}  {attempt:3}  {0:6}  {niter:5}  {niter:5}  {timer.time1:14.6f}  {timer.time1:14.6f}  {timer.dtime:14.6f}')

                self.assembly.update_element_data()
                self.assembly.dof_solution += self.assembly.ddof_solution
                # 调换了上面两行代码的顺序，基于t时刻的自由度值及t+dt时刻的自由度增量值对单元信息进行更新，之后在将所有单元的自由度值更新为t+dt时刻。

                self.assembly.update_element_state_variables()
                self.assembly.update_element_field_variables()
                self.assembly.assembly_field_variables()

                # time0 = time.time()
                self.database.add_hdf5()
                for output in self.assembly.props.outputs:
                    if output.is_save:
                        if output.type == 'vtk':
                            write_vtk(self.assembly)
                        if output.type == 'hdf5':
                            add_hdf5(self.assembly)
                # time1 = time.time()
                # print('write_vtk: ', time1 - time0)

                timer.time0 = timer.time1
                timer.frame_ids.append(increment)
                increment += 1
                attempt = 1
                timer.dtime *= 1.1
                if timer.dtime >= self.solver.max_dtime:
                    timer.dtime = self.solver.max_dtime
                if timer.time0 + timer.dtime >= self.solver.total_time:
                    timer.dtime = self.solver.total_time - timer.time0

            else:
                attempt += 1
                timer.dtime *= 0.5
                logger.warning(f'  increment {increment} is divergence, dtime is reduced to {timer.dtime}')

                if timer.dtime <= self.assembly.props.solver.min_dtime:
                    for output in self.assembly.props.outputs:
                        if output.is_save:
                            if output.type == 'vtk':
                                write_pvd(self.assembly)
                    logger.error(f'Computation is ended with error: the dtime {timer.dtime} is less than the minimum value')
                    return -1

                self.assembly.ddof_solution = zeros(self.assembly.total_dof_number, dtype=DTYPE)
                self.assembly.goback_element_state_variables()
                self.assembly.update_element_data()
                self.assembly.assembly_fint()

            if timer.is_done():
                for output in self.assembly.props.outputs:
                    if output.is_save:
                        if output.type == 'vtk':
                            write_pvd(self.assembly)
                break

        if not timer.is_done():
            logger.error('maximum increment is reached')
            return -1
        else:
            return 0


if __name__ == "__main__":
    from pyfem.utils.visualization import print_slots_dict

    print_slots_dict(NonlinearSolver.__slots_dict__)

    from pyfem.Job import Job

    import numpy as np
    np.set_printoptions(precision=5, suppress=True, linewidth=10000)

    job = Job(r'..\..\..\examples\mechanical\beam\Job-1.toml')
    # job = Job(r'..\..\..\examples\mechanical\1element\quad4\Job-1.toml')
    # solver = NonlinearSolver(job.assembly, job.props.solver)
    # solver.show()
    job.run()
