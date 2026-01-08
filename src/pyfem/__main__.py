# -*- coding: utf-8 -*-
"""

"""
import traceback
from pathlib import Path

from pyfem import __version__
from pyfem.job.Job import Job
from pyfem.io.arguments import get_arguments
from pyfem.utils.logger import logger, set_logger, logger_sta, set_logger_sta
from pyfem.utils.wrappers import show_running_time
from pyfem.parallel.mpi_setup import get_mpi_context
from pyfem.fem.constants import IS_PETSC


@show_running_time
def main() -> None:
    if IS_PETSC:
        main_mpi()
    else:
        main_serial()


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


def main_mpi() -> None:
    # 初始化MPI环境
    mpi_ctx = get_mpi_context()
    comm = mpi_ctx['comm']
    rank = mpi_ctx['rank']

    # 解析命令行参数，所有进程都需要
    args = get_arguments()
    input_file = Path(args.i)

    # 确保所有进程在继续前都已解析参数
    comm.Barrier()

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
        job = Job(args.i)
        job.run()
        # import time
        # time.sleep(10)  # 确保日志完整写入
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
