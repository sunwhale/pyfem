# -*- coding: utf-8 -*-
"""

"""
from pyfem.Job import Job
from pyfem.io.arguments import get_arguments
from pyfem.utils.wrappers import show_running_time


@show_running_time
def main() -> None:
    args = get_arguments()

    job = Job(args.i)

    job.run()
