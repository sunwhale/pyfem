# -*- coding: utf-8 -*-
"""
定义一些导入模块的辅助函数
"""

from pyfem.utils.colors import error_style


def import_petsc4py():
    try:
        from petsc4py import PETSc  # type: ignore
        return PETSc
    except ImportError:
        raise ImportError(error_style('petsc4py can not be imported'))


def import_mpi4py():
    try:
        from mpi4py import MPI  # type: ignore
        return MPI
    except ImportError:
        raise ImportError(error_style("mpi4py can not be imported."))
