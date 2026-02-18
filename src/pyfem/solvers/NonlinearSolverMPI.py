# -*- coding: utf-8 -*-
"""

"""
from copy import deepcopy

import numpy as np
import scipy as sp  # type: ignore

from pyfem.assembly.Assembly import Assembly
from pyfem.database.Database import Database
from pyfem.fem.constants import DTYPE, IS_PETSC, IS_MPI
from pyfem.io.Solver import Solver
from pyfem.io.write_vtk import write_vtk, write_pvd
from pyfem.parallel.mpi_setup import get_mpi_context
from pyfem.solvers.BaseSolver import BaseSolver
from pyfem.utils.colors import error_style
from pyfem.utils.logger import logger, logger_sta
from pyfem.utils.wrappers import show_running_time

if IS_PETSC:
    try:
        from petsc4py import PETSc  # type: ignore
    except:
        raise ImportError(error_style('petsc4py can not be imported'))


class NonlinearSolver(BaseSolver):
    r"""
    非线性求解器。

    :ivar is_convergence: 是否收敛
    :vartype is_convergence: bool

    :ivar increment: 增量步
    :vartype increment: int

    :ivar niter: 迭代步
    :vartype niter: int

    :ivar attempt: 尝试步
    :vartype attempt: int

    :ivar f_residual: 残差
    :vartype f_residual: float

    :ivar fint: 内力向量
    :vartype fint: np.ndarray(total_dof_number,)

    :ivar rhs: 等式右边向量
    :vartype rhs: np.ndarray(total_dof_number,)

    :ivar b: 等式右边向量
    :vartype b: petsc4py.PETSc.Vec(total_dof_number)

    :ivar x: 解向量
    :vartype x: petsc4py.PETSc.Vec(total_dof_number)

    :ivar da: 解向量
    :vartype da: np.ndarray(total_dof_number,)

    :ivar mpi_context: MPI上下文字典
    :vartype mpi_context: MPIContext

    :ivar comm: MPI通信器
    :vartype comm: MPI.Comm

    :ivar rank: MPI进程编号
    :vartype rank: int

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
        'is_convergence': ('bool', '是否收敛'),
        'increment': ('int', '增量步'),
        'niter': ('int', '迭代步'),
        'attempt': ('int', '尝试步'),
        'f_residual': ('float', '残差'),
        'fint': ('np.ndarray(total_dof_number,)', '内力向量'),
        'rhs': ('np.ndarray(total_dof_number,)', '等式右边向量'),
        'b': ('petsc4py.PETSc.Vec(total_dof_number)', '等式右边向量'),
        'x': ('petsc4py.PETSc.Vec(total_dof_number)', '解向量'),
        'da': ('np.ndarray(total_dof_number,)', '解向量'),
        'mpi_context': ('MPIContext', 'MPI上下文字典'),
        'comm': ('MPI.Comm', 'MPI通信器'),
        'rank': ('int', 'MPI进程编号'),
        'PENALTY': ('float', '罚系数'),
        'FORCE_TOL': ('float', '残差容限'),
        'MAX_NITER': ('int', '最大迭代次数'),
        'BC_METHOD': ('str', '边界条件施加方式')
    }

    __slots__ = BaseSolver.__slots__ + [slot for slot in __slots_dict__.keys()]

    def __init__(self, assembly: Assembly, solver: Solver) -> None:
        super().__init__()
        self.is_convergence: bool = False
        self.increment: int = 0
        self.niter: int = 0
        self.attempt: int = 0
        self.f_residual: float = 0.0
        self.assembly: Assembly = assembly
        self.solver: Solver = solver
        self.dof_solution: np.ndarray = np.zeros(self.assembly.total_dof_number)
        self.database: Database = Database(self.assembly)
        self.fint: np.ndarray = np.empty(0, dtype=DTYPE)
        self.rhs: np.ndarray = np.empty(0, dtype=DTYPE)
        self.da: np.ndarray = np.empty(0, dtype=DTYPE)
        self.mpi_context = get_mpi_context()
        self.comm = self.mpi_context['comm']
        self.rank: int = self.mpi_context['rank']

        if IS_PETSC and IS_MPI:
            self.b: PETSc.Vec = PETSc.Vec().create(comm=self.comm)
            self.b.setSizes(self.assembly.total_dof_number)
            self.b.setUp()

            self.x: PETSc.Vec = PETSc.Vec().create(comm=self.comm)
            self.x.setSizes(self.assembly.total_dof_number)
            self.x.setUp()

        elif IS_PETSC and not IS_MPI:
            self.b: PETSc.Vec = PETSc.Vec()
            self.x: PETSc.Vec = PETSc.Vec()

        else:
            self.b = None
            self.x = None

        self.PENALTY: float = 1.0e128
        self.FORCE_TOL: float = 1.0e-3
        self.MAX_NITER: int = 8
        self.BC_METHOD: str = '01'

    def run(self) -> int:
        if self.assembly.props.solver.option in [None, '', 'NR', 'NewtonRaphson']:
            return self.incremental_iterative_solve('NR')
        elif self.assembly.props.solver.option in ['IT', 'InitialTangent']:
            return self.incremental_iterative_solve('IT')
        else:
            raise NotImplementedError(error_style(f'unsupported option \'{self.assembly.props.solver.option}\' of {self.assembly.props.solver.type}'))

    # @show_running_time
    def apply_bcs(self) -> None:
        timer = self.assembly.timer
        # 罚系数法施加边界条件
        if self.BC_METHOD == 'PENALTY':
            if self.niter == 0:
                for bc_data in self.assembly.bc_data_list:
                    amplitude_increment = bc_data.get_amplitude(timer.time1) - bc_data.get_amplitude(timer.time0)
                    if bc_data.bc.category == 'DirichletBC':
                        for bc_dof_id, bc_dof_value in zip(bc_data.bc_dof_ids, bc_data.bc_dof_values):
                            self.assembly.global_stiffness[bc_dof_id, bc_dof_id] += self.PENALTY
                            self.rhs[bc_dof_id] += bc_dof_value * self.PENALTY * amplitude_increment
                    elif bc_data.bc.category == 'NeumannBC':
                        for bc_dof_id, bc_fext in zip(bc_data.bc_dof_ids, bc_data.bc_fext):
                            self.rhs[bc_dof_id] += bc_fext * amplitude_increment
                            self.assembly.fext[bc_dof_id] += bc_fext * amplitude_increment
            else:
                for bc_data in self.assembly.bc_data_list:
                    if bc_data.bc.category == 'DirichletBC':
                        for bc_dof_id, bc_dof_value in zip(bc_data.bc_dof_ids, bc_data.bc_dof_values):
                            self.assembly.global_stiffness[bc_dof_id, bc_dof_id] += self.PENALTY
                            self.rhs[bc_dof_id] = 0.0 * self.PENALTY

        # 划0置1法
        if self.BC_METHOD == '01':

            if IS_PETSC and IS_MPI:
                if self.rank == 0:
                    self.b.setValues(range(self.assembly.total_dof_number), self.assembly.fext)
                self.comm.barrier()
                self.b.assemble()

            elif IS_PETSC and not IS_MPI:
                self.b = self.assembly.A.createVecLeft()
                self.x = self.assembly.A.createVecRight()
                self.b.setValues(range(self.assembly.total_dof_number), self.assembly.fext)

            else:
                self.assembly.global_stiffness = self.assembly.global_stiffness.tolil()
                pass

            if self.niter == 0:
                for bc_data in self.assembly.bc_data_list:
                    amplitude_increment = bc_data.get_amplitude(timer.time1) - bc_data.get_amplitude(timer.time0)
                    if bc_data.bc.category == 'DirichletBC':
                        if IS_PETSC and IS_MPI:
                            if self.rank == 0:
                                self.x.setValues(bc_data.bc_dof_ids, bc_data.bc_dof_values * amplitude_increment)
                            self.comm.barrier()
                            self.x.assemble()

                            self.assembly.A.zeroRowsColumns(bc_data.bc_dof_ids, diag=1.0, x=self.x, b=self.b)
                            self.comm.barrier()
                            self.assembly.A.assemble()

                            if self.rank == 0:
                                self.b.setValues(bc_data.bc_dof_ids, bc_data.bc_dof_values * amplitude_increment + self.fint[bc_data.bc_dof_ids])
                            self.comm.barrier()
                            self.b.assemble()

                        elif IS_PETSC and not IS_MPI:
                            self.x.setValues(bc_data.bc_dof_ids, bc_data.bc_dof_values * amplitude_increment)
                            self.assembly.A.zeroRowsColumns(bc_data.bc_dof_ids, diag=1.0, x=self.x, b=self.b)
                            self.b.setValues(bc_data.bc_dof_ids, bc_data.bc_dof_values * amplitude_increment + self.fint[bc_data.bc_dof_ids])

                        else:
                            # 注意此处的乘法为lil_matrix与ndarray相乘，其广播方式不同
                            self.rhs -= self.assembly.global_stiffness[:, bc_data.bc_dof_ids] * bc_data.bc_dof_values * amplitude_increment
                            self.rhs[bc_data.bc_dof_ids] = bc_data.bc_dof_values * amplitude_increment + self.fint[bc_data.bc_dof_ids]
                            self.assembly.global_stiffness[bc_data.bc_dof_ids, :] = 0.0
                            self.assembly.global_stiffness[:, bc_data.bc_dof_ids] = 0.0
                            self.assembly.global_stiffness[bc_data.bc_dof_ids, bc_data.bc_dof_ids] = 1.0
                            # for bc_dof_id, bc_dof_value in zip(bc_data.bc_dof_ids, bc_data.bc_dof_values):
                            #     rhs -= self.assembly.global_stiffness[:, bc_dof_id].toarray().reshape(-1) * bc_dof_value * amplitude_increment
                            #     self.assembly.global_stiffness[bc_dof_id, :] = 0.0
                            #     self.assembly.global_stiffness[:, bc_dof_id] = 0.0
                            #     self.assembly.global_stiffness[bc_dof_id, bc_dof_id] = 1.0
                            #     rhs[bc_dof_id] = bc_dof_value * amplitude_increment + fint[bc_dof_id]

                    elif bc_data.bc.category == 'NeumannBC':
                        if IS_PETSC and IS_MPI:
                            if self.rank == 0:
                                self.b.setValues(bc_data.bc_dof_ids, bc_data.bc_fext * amplitude_increment, addv=True)
                            self.comm.barrier()
                            self.b.assemble()

                            self.assembly.fext[bc_data.bc_dof_ids] += bc_data.bc_fext * amplitude_increment
                            self.comm.barrier()
                            self.assembly.A.assemble()

                        elif IS_PETSC and not IS_MPI:
                            self.b.setValues(bc_data.bc_dof_ids, bc_data.bc_fext * amplitude_increment, addv=True)
                            self.assembly.fext[bc_data.bc_dof_ids] += bc_data.bc_fext * amplitude_increment

                        else:
                            for bc_dof_id, bc_fext in zip(bc_data.bc_dof_ids, bc_data.bc_fext):
                                self.rhs[bc_dof_id] += bc_fext * amplitude_increment
                                self.assembly.fext[bc_dof_id] += bc_fext * amplitude_increment

            else:
                for bc_data in self.assembly.bc_data_list:
                    if bc_data.bc.category == 'DirichletBC':
                        if IS_PETSC and IS_MPI:
                            if self.rank == 0:
                                self.b.setValues(bc_data.bc_dof_ids, self.fint[bc_data.bc_dof_ids])
                            self.comm.barrier()
                            self.b.assemble()

                            self.assembly.A.zeroRowsColumns(bc_data.bc_dof_ids)
                            self.comm.barrier()
                            self.assembly.A.assemble()

                        elif IS_PETSC and not IS_MPI:
                            self.b.setValues(bc_data.bc_dof_ids, self.fint[bc_data.bc_dof_ids])
                            self.assembly.A.zeroRowsColumns(bc_data.bc_dof_ids)

                        else:
                            self.assembly.global_stiffness[bc_data.bc_dof_ids, :] = 0.0
                            self.assembly.global_stiffness[:, bc_data.bc_dof_ids] = 0.0
                            self.assembly.global_stiffness[bc_data.bc_dof_ids, bc_data.bc_dof_ids] = 1.0
                            self.rhs[bc_data.bc_dof_ids] = self.fint[bc_data.bc_dof_ids]
                            # for bc_dof_id, bc_dof_value in zip(bc_data.bc_dof_ids, bc_data.bc_dof_values):
                            #     self.assembly.global_stiffness[bc_dof_id, :] = 0.0
                            #     self.assembly.global_stiffness[:, bc_dof_id] = 0.0
                            #     self.assembly.global_stiffness[bc_dof_id, bc_dof_id] = 1.0
                            #     rhs[bc_dof_id] = fint[bc_dof_id]

    def gather_vector_to_rank0(self, vector):
        """将分布式向量收集到rank 0进程"""
        # 创建串行向量用于收集
        scatter, v_seq = PETSc.Scatter.toZero(vector)
        scatter.scatter(vector, v_seq, False, PETSc.Scatter.Mode.FORWARD)

        # 在rank 0上转换为numpy数组
        if self.rank == 0:
            array = v_seq.getArray().copy()
        else:
            array = None

        # 清理临时对象
        scatter.destroy()
        v_seq.destroy()

        return array

    # @show_running_time
    def solve_linear_system(self) -> bool:
        try:
            if IS_PETSC and IS_MPI:

                self.comm.barrier()
                self.assembly.A.assemble()

                ksp = PETSc.KSP().create(comm=self.comm)
                ksp.setOperators(self.assembly.A)

                if self.rank == 0:
                    self.b.setValues(range(self.assembly.total_dof_number), -self.fint, addv=True)
                self.comm.barrier()
                self.b.assemble()

                # 直接求解
                # ksp.setType('preonly')
                # ksp.getPC().setType('lu')
                # ksp.setConvergenceHistory()

                # 迭代求解
                ksp.setType('bcgs')
                ksp.getPC().setType('sor')
                ksp.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)
                ksp.setConvergenceHistory()

                ksp.solve(self.b, self.x)
                self.da = self.gather_vector_to_rank0(self.x)

            elif IS_PETSC and not IS_MPI:
                self.assembly.A.assemble()
                ksp = PETSc.KSP().create()
                ksp.setOperators(self.assembly.A)
                self.b.setValues(range(self.assembly.total_dof_number), -self.fint, addv=True)

                # 直接求解
                # ksp.setType('preonly')
                # ksp.getPC().setType('lu')
                # ksp.setConvergenceHistory()

                # 迭代求解
                ksp.setType('bcgs')
                ksp.getPC().setType('sor')
                ksp.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)
                ksp.setConvergenceHistory()

                ksp.solve(self.b, self.x)

                self.da = self.x.array[:]
            else:
                self.assembly.global_stiffness = self.assembly.global_stiffness.tocsc()
                LU = sp.sparse.linalg.splu(self.assembly.global_stiffness)
                self.da = LU.solve(self.rhs - self.fint)
            return True

        except RuntimeError as e:
            self.is_convergence = False
            print(error_style(f"Catch RuntimeError exception: {e}"))
            return False

    # @show_running_time
    def write_database_initiation(self):
        self.database.init_hdf5()
        for output in self.assembly.props.outputs:
            if output.is_save:
                if output.type == 'vtk':
                    write_vtk(self.assembly)

    # @show_running_time
    def write_database_frame(self):
        self.database.add_hdf5()
        for output in self.assembly.props.outputs:
            if output.is_save:
                if output.type == 'vtk':
                    write_vtk(self.assembly)

    def write_database_done(self):
        for output in self.assembly.props.outputs:
            if output.is_save:
                if output.type == 'vtk':
                    write_pvd(self.assembly)

    def timer_initiation(self):
        self.assembly.timer.total_time = self.solver.total_time
        self.assembly.timer.dtime = self.solver.initial_dtime
        self.assembly.timer.time0 = self.solver.start_time
        self.assembly.timer.increment = 0
        self.assembly.timer.frame_ids.append(0)

    def timer_increment(self):
        self.increment += 1
        self.attempt = 1
        self.assembly.timer.time0 = self.assembly.timer.time1
        self.assembly.timer.frame_ids.append(self.increment)
        self.assembly.timer.dtime *= 1.1
        if self.assembly.timer.dtime >= self.solver.max_dtime:
            self.assembly.timer.dtime = self.solver.max_dtime
        if self.assembly.timer.time0 + self.assembly.timer.dtime >= self.solver.total_time:
            self.assembly.timer.dtime = self.solver.total_time - self.assembly.timer.time0

    def get_convergence(self) -> bool:
        self.f_residual = self.assembly.fext - self.assembly.fint
        self.f_residual[self.assembly.bc_dof_ids] = 0
        if np.linalg.norm(self.assembly.fext) < 1.0e-16:
            self.f_residual = np.linalg.norm(self.f_residual)
        else:
            self.f_residual = np.linalg.norm(self.f_residual) / np.linalg.norm(self.assembly.fext)
        # self.f_residual = max(abs(self.f_residual))

        logger.log(21, f'  niter = {self.niter}, residual = {self.f_residual}')

        if self.f_residual < self.FORCE_TOL:
            self.is_convergence = True
            return True
        else:
            self.is_convergence = False
            return False

    def incremental_iterative_solve(self, option: str) -> int:
        self.increment: int = 1
        self.attempt: int = 1
        self.timer_initiation()
        if self.rank == 0:
            self.assembly.update_element_field_variables()
            self.assembly.assembly_field_variables()
            self.write_database_initiation()
        timer = self.assembly.timer

        for i in range(1, self.solver.max_increment):
            timer.time1 = timer.time0 + timer.dtime
            timer.increment = self.increment
            logger.info(f'increment = {self.increment}, attempt = {self.attempt}, time = {timer.time1:14.9f}, dtime = {timer.dtime:14.9f}')
            if self.rank == 0:
                self.assembly.ddof_solution = np.zeros(self.assembly.total_dof_number, dtype=DTYPE)
                self.assembly.update_element_data()
            self.is_convergence = False
            for self.niter in range(self.MAX_NITER):
                # print(f"进程 {self.rank} 装配全局刚度矩阵")
                self.assembly.assembly_global_stiffness()
                if self.rank == 0:
                    self.fint = deepcopy(self.assembly.fint)
                    self.rhs = deepcopy(self.assembly.fext)
                self.apply_bcs()

                if not self.solve_linear_system():
                    break

                if self.rank == 0:
                    self.assembly.ddof_solution += self.da
                    if option == 'NR':
                        self.assembly.update_element_data()
                    elif option == 'IT':
                        self.assembly.update_element_data_without_stiffness()
                    self.assembly.assembly_fint()

                if timer.is_reduce_dtime:  # 本构方程中局部迭代不收敛，可能触发该事件
                    timer.is_reduce_dtime = False
                    self.is_convergence = False
                    break

                if self.rank == 0:
                    self.is_convergence = self.get_convergence()

                self.is_convergence = self.comm.bcast(self.is_convergence, root=0)
                if self.is_convergence:
                    break

            if self.is_convergence:

                if self.rank == 0:
                    logger.info(f'  increment {self.increment} is convergence')
                    logger_sta.info(f'{1:4}  {self.increment:9}  {self.attempt:3}  {0:6}  {self.niter:5}  {self.niter:5}  {timer.time1:14.6f}  {timer.time1:14.6f}  {timer.dtime:14.6f}')

                    # 注意下面标记的两行代码顺序
                    self.assembly.update_element_data()  # 基于t时刻的<自由度值>及t+dt时刻的<自由度增量值>对单元信息进行更新
                    self.assembly.update_element_state_variables()
                    self.assembly.update_element_field_variables()
                    self.assembly.assembly_field_variables()
                    self.write_database_frame()

                    self.assembly.dof_solution += self.assembly.ddof_solution  # 将所有单元的<自由度值>更新为t+dt时刻
                self.timer_increment()
            else:
                self.attempt += 1
                timer.dtime *= 0.5

                if self.rank == 0:
                    logger.warning(f'  increment {self.increment} is divergence, dtime is reduced to {timer.dtime}')

                    if timer.dtime <= self.assembly.props.solver.min_dtime:
                        self.write_database_done()
                        logger.error(f'Computation is ended with error: the dtime {timer.dtime} is less than the minimum value')
                        return -1

                    self.assembly.ddof_solution = np.zeros(self.assembly.total_dof_number, dtype=DTYPE)
                    self.assembly.goback_element_state_variables()
                    self.assembly.update_element_data()
                    self.assembly.assembly_fint()

            if timer.is_done():
                self.write_database_done()
                break

        if not timer.is_done():
            logger.error('maximum increment is reached')
            return -1
        else:
            return 0


if __name__ == "__main__":
    from pyfem.utils.visualization import print_slots_dict

    print_slots_dict(NonlinearSolver.__slots_dict__)

    from pyfem.job.Job import Job

    import numpy as np

    np.set_printoptions(precision=5, suppress=True, linewidth=10000)

    # job = Job(r'..\..\..\examples\mechanical\beam\Job-1.toml')
    job = Job(r'..\..\..\examples\mechanical\plane\Job-1.toml')
    # job = Job(r'..\..\..\examples\mechanical\1element\quad4\Job-1.toml')
    solver = NonlinearSolver(job.assembly, job.props.solver)
    solver.show()
    job.assembly.show()
    job.run()
