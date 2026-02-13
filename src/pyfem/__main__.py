# -*- coding: utf-8 -*-
"""

"""
import traceback
from pathlib import Path

from pyfem import __version__
from pyfem.fem.constants import IS_PETSC, IS_MPI
from pyfem.utils.colors import error_style

if IS_MPI:
    try:
        from mpi4py import MPI  # type: ignore
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            error_style("Parallel version requires mpi4py.\n"
                        "Please install it or use the serial version.")
        )

if IS_PETSC:
    try:
        from petsc4py import PETSc  # type: ignore
        from petsc4py.PETSc import Mat  # type: ignore
    except ModuleNotFoundError:
        raise ModuleNotFoundError(error_style('petsc4py can not be imported'))

from pyfem.io.arguments import get_arguments
from pyfem.job.Job import Job
from pyfem.job.JobMPI import JobMPI
from pyfem.parallel.mpi_setup import get_mpi_context
from pyfem.utils.logger import logger, set_logger, logger_sta, set_logger_sta
from pyfem.utils.wrappers import show_running_time


@show_running_time
def main() -> None:
    if IS_MPI:
        main_mpi()
    else:
        main_serial()


@show_running_time
def main_serial() -> None:
    args = get_arguments()

    input_file = Path(args.i)

    if input_file.is_absolute():
        abs_input_file = input_file
    else:
        abs_input_file = Path.cwd().joinpath(input_file).resolve()

    set_logger(logger, abs_input_file)
    set_logger_sta(logger_sta, abs_input_file)

    logger.info(f'ANALYSIS INITIATED FROM PYFEM {__version__}')

    lock_file = abs_input_file.with_suffix('.lck')

    if lock_file.exists():
        exit(f'Error: The job {abs_input_file} is locked.\nIt may be running or terminated with exception.')

    lock_file.touch()

    try:
        job = Job(args.i)
        job.run()
    except KeyboardInterrupt:
        logger.error('JOB EXITED WITH KEYBOARD INTERRUPT')
        logger_sta.error('THE ANALYSIS HAS NOT BEEN COMPLETED')
    except Exception as e:
        traceback.print_exc()
        logger.error(e)
        logger.error('JOB EXITED WITH ERROR')
        logger_sta.error('THE ANALYSIS HAS NOT BEEN COMPLETED')

    finally:
        try:
            if 'lock_file' in locals():
                lock_file.unlink()
        except Exception:
            pass  # 忽略锁文件删除错误


@show_running_time
def main_mpi() -> None:
    # 初始化MPI环境
    mpi_context = get_mpi_context()
    comm = mpi_context['comm']
    rank = mpi_context['rank']

    # 解析命令行参数，所有进程都需要
    args = get_arguments()
    input_file = Path(args.i)

    # 确保所有进程在继续前都已解析参数
    comm.barrier()

    if rank == 0:
        # 只有主进程处理文件操作和全局控制
        if input_file.is_absolute():
            abs_input_file = input_file
        else:
            abs_input_file = Path.cwd().joinpath(input_file).resolve()

        set_logger(logger, abs_input_file)
        set_logger_sta(logger_sta, abs_input_file)

        logger.info(f'ANALYSIS INITIATED FROM PYFEM {__version__}')

        lock_file = abs_input_file.with_suffix('.lck')

        if lock_file.exists():
            # 广播错误给所有进程并退出
            error_msg = f'Error: The job {abs_input_file} is locked.\nIt may be running or terminated with exception.'
            logger.error(error_msg)
            comm.bcast(('ERROR', error_msg), root=0)
            comm.Abort(1)

        lock_file.touch()

    try:
        job = JobMPI(args.i)
        job.run()
    except KeyboardInterrupt:
        if rank == 0:
            logger.error('JOB EXITED WITH KEYBOARD INTERRUPT')
            logger_sta.error('THE ANALYSIS HAS NOT BEEN COMPLETED')
            comm.bcast(('INTERRUPT', 'JOB EXITED WITH KEYBOARD INTERRUPT'), root=0)
    except Exception as e:
        if rank == 0:
            traceback.print_exc()
            logger.error(e)
            logger.error('JOB EXITED WITH ERROR')
            logger_sta.error('THE ANALYSIS HAS NOT BEEN COMPLETED')
            comm.bcast(('ERROR', 'JOB EXITED WITH ERROR'), root=0)

    finally:
        if rank == 0:
            # 只有主进程清理锁文件
            try:
                if 'lock_file' in locals():
                    lock_file.unlink()
            except Exception:
                pass  # 忽略锁文件删除错误
