# -*- coding: utf-8 -*-
"""

"""
from pathlib import Path

from pyfem.Job import Job
from pyfem.io.arguments import get_arguments
from pyfem.utils.logger import logger, set_logger, logger_sta, set_logger_sta
from pyfem.utils.wrappers import show_running_time


@show_running_time
def main() -> None:
    args = get_arguments()

    input_file = Path(args.i)

    if input_file.is_absolute():
        abs_input_file = input_file
    else:
        abs_input_file = Path.cwd().joinpath(input_file).resolve()

    set_logger(logger, abs_input_file)
    set_logger_sta(logger_sta, abs_input_file)

    lock_file = abs_input_file.with_suffix('.lck')

    if lock_file.exists():
        exit(f'Error: The job {abs_input_file} is locked.\nIt may be running or terminated with exception.')

    lock_file.touch()

    try:
        job = Job(args.i)
        job.run()
    except Exception as e:
        logger.error(e)
        logger.error('JOB EXITED')
        logger_sta.error('THE ANALYSIS HAS NOT BEEN COMPLETED')

    lock_file.unlink()
