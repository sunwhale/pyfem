# -*- coding: utf-8 -*-
"""

"""
from pathlib import Path
from typing import Union

from pyfem.assembly.Assembly import Assembly
from pyfem.io.Properties import Properties
from pyfem.solvers.get_solver_data import get_solver_data, SolverData
from pyfem.utils.logger import logger, logger_sta, STA_HEADER
from pyfem.utils.visualization import object_slots_to_string
from pyfem.parallel.mpi_setup import get_mpi_context


class MPIJob:
    """
    求解器基类。

    :ivar input_file: 输入文件路径
    :vartype input_file: Path

    :ivar work_directory: 工作目录
    :vartype work_directory: Path

    :ivar abs_input_file: 输入文件绝对路径
    :vartype abs_input_file: Path

    :ivar props: 属性对象
    :vartype props: Properties

    :ivar assembly: 装配体属性
    :vartype assembly: Assembly

    :ivar solver_data: 求解器对象
    :vartype solver_data: SolverData
    """

    __slots_dict__: dict = {
        'input_file': ('Path', '输入文件路径'),
        'work_directory': ('Path', '工作目录'),
        'abs_input_file': ('Path', '输入文件绝对路径'),
        'props': ('Properties', '属性对象'),
        'assembly': ('Assembly', '装配体属性'),
        'solver_data': ('SolverData', '求解器对象')
    }

    __slots__: list = [slot for slot in __slots_dict__.keys()]

    def __init__(self, filename: Union[Path, str]) -> None:
        # 初始化MPI环境
        mpi_context = get_mpi_context()
        comm = mpi_context['comm']
        rank = mpi_context['rank']

        input_file = Path(filename)
        if input_file.is_absolute():
            abs_input_file = input_file
        else:
            abs_input_file = Path.cwd().joinpath(input_file).resolve()
        self.input_file: Path = input_file
        self.work_directory: Path = Path.cwd()
        self.abs_input_file: Path = abs_input_file

        comm.Barrier()

        if rank == 0:
            self.props: Properties = Properties()
            self.props.read_file(abs_input_file)
            self.assembly: Assembly = Assembly(self.props)
            # self.solver_data: SolverData = get_solver_data(self.assembly, self.props.solver)

        else:
            self.props: Properties = Properties()
            self.props.read_file(abs_input_file)
            self.assembly: Assembly = Assembly(self.props)
            # self.solver_data: SolverData = get_solver_data(self.assembly, self.props.solver)

    def run(self) -> int:
        logger.info(f'SOLVER RUNNING')
        logger_sta.info(STA_HEADER)
        status = self.solver_data.run()
        if status == 0:
            logger.info(f'JOB COMPLETED')
            logger_sta.info('THE ANALYSIS HAS COMPLETED SUCCESSFULLY')
        else:
            logger.warning(f'JOB EXITED')
            logger_sta.warning('THE ANALYSIS HAS NOT BEEN COMPLETED')
        return status

    def run_with_log(self) -> int:
        status = -1
        return status

    def to_string(self, level: int = 1) -> str:
        return object_slots_to_string(self, level)

    def show(self) -> None:
        print(self.to_string())


if __name__ == '__main__':
    from pyfem.utils.visualization import print_slots_dict

    print_slots_dict(ParallelJob.__slots_dict__)
