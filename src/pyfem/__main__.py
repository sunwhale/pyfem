# -*- coding: utf-8 -*-
"""

"""
from pyfem.Job import Job
from pyfem.io.arguments import get_arguments
from pyfem.utils.wrappers import show_running_time
from pyfem.utils.logger import logger, set_logger, logger_sta, set_logger_sta, STA_HEADER


@show_running_time
def main() -> None:
    args = get_arguments()

    job = Job(args.i)

    set_logger(logger, job=job)
    set_logger_sta(logger_sta, job=job)

    try:
        logger.info(f'SOLVER RUNNING')
        logger_sta.info(STA_HEADER)
        status = job.run()
        if status:
            logger.info(f'JOB COMPLETED')
            logger_sta.info('THE ANALYSIS HAS COMPLETED SUCCESSFULLY')
        else:
            logger.warning(f'JOB EXITED')
            logger_sta.warning('THE ANALYSIS HAS NOT BEEN COMPLETED')
    except:
        logger.error('JOB EXITED')
        logger_sta.error('THE ANALYSIS HAS NOT BEEN COMPLETED')
