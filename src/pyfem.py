# -*- coding: utf-8 -*-
"""

"""
from pyfem.__main__ import main_serial, main_mpi
from pyfem.fem.constants import IS_PETSC

if IS_PETSC:
    main_mpi()
else:
    main_serial()
