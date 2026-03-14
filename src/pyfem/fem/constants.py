# -*- coding: utf-8 -*-
"""
定义常数
"""
from pyfem.io.arguments import get_arguments

DTYPE = 'float64'

LOGO = r"""
                 ____             
    ____  __  __/ __/__  ____ ___ 
   / __ \/ / / / /_/ _ \/ __ `__ \
  / /_/ / /_/ / __/  __/ / / / / /
 / .___/\__, /_/  \___/_/ /_/ /_/ 
/_/    /____/                     
"""

_args = get_arguments()

IS_MPI = _args.mpi

IS_PETSC = _args.petsc

IS_DEBUG = _args.debug
