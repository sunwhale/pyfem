# -*- coding: utf-8 -*-
"""

"""
import sys
from typing import Dict, Any, Optional, Union

try:
    from mpi4py import MPI

    MPIContext = Union[MPI.Comm, int, bool]
except ImportError:
    MPIContext = None


def setup_mpi() -> Dict[str, MPIContext]:
    """Initialize MPI environment and return context dictionary."""
    try:
        from mpi4py import MPI
        from petsc4py import PETSc

        comm: MPI.Comm = MPI.COMM_WORLD
        rank: int = comm.Get_rank()
        size: int = comm.Get_size()

        return {
            'comm': comm,
            'rank': rank,
            'size': size,
            'is_parallel': size > 1,
            'is_master': rank == 0,
            'is_worker': rank > 0
        }
    except ImportError as e:
        raise RuntimeError(
            "Parallel version requires mpi4py and petsc4py.\n"
            "Please install them or use the serial version."
        ) from e


# MPI context singleton
_MPI_CONTEXT: Optional[Dict[str, MPIContext]] = None


def get_mpi_context() -> Dict[str, MPIContext]:
    """Get MPI context (singleton pattern)."""
    global _MPI_CONTEXT
    if _MPI_CONTEXT is None:
        try:
            _MPI_CONTEXT = setup_mpi()
        except RuntimeError:
            # 在串行模式下提供模拟的MPI上下文
            class DummyComm:
                """模拟的MPI通信器"""
                _is_dummy_comm = True

                @staticmethod
                def Get_rank():
                    return 0

                @staticmethod
                def Get_size():
                    return 1

                @staticmethod
                def Barrier():
                    pass

                @staticmethod
                def bcast(obj, root=0):
                    return obj

                @staticmethod
                def Abort(code=1):
                    sys.exit(code)

            _MPI_CONTEXT = {
                'comm': DummyComm(),
                'rank': 0,
                'size': 1,
                'is_parallel': False,
                'is_master': True,
                'is_worker': False
            }
    return _MPI_CONTEXT


def is_dummy_comm(comm):
    """判断通信器是否为DummyComm"""
    return getattr(comm, '_is_dummy_comm', False)


if __name__ == "__main__":
    mpi_info = get_mpi_context()
    print(f"MPI Rank: {mpi_info['rank']}, Size: {mpi_info['size']}, Is Parallel: {mpi_info['is_parallel']}")
