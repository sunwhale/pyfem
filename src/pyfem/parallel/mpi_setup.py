"""MPI/PETSc初始化模块 - 只在并行版本导入时执行"""


def setup_mpi():
    """初始化MPI环境"""
    try:
        from mpi4py import MPI
        from petsc4py import PETSc

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        # 可选：初始化PETSc
        # PETSc.Sys.popErrorHandler()  # 禁用错误处理器
        # 或者 PETSc.Options().setValue(...)

        return {
            'comm': comm,
            'rank': rank,
            'size': size,
            'is_parallel': size > 1
        }
    except ImportError as e:
        raise RuntimeError(
            "并行版本需要mpi4py和petsc4py。请安装或使用串行版本。"
        ) from e


def get_mpi_info():
    """获取MPI信息（惰性初始化）"""
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    return comm.Get_rank(), comm.Get_size()


# 全局变量（按需使用）
_MPI_CONTEXT = None


def get_mpi_context():
    """获取或创建MPI上下文（单例模式）"""
    global _MPI_CONTEXT
    if _MPI_CONTEXT is None:
        _MPI_CONTEXT = setup_mpi()
    return _MPI_CONTEXT


if __name__ == "__main__":
    mpi_info = get_mpi_context()
    print(f"MPI Rank: {mpi_info['rank']}, Size: {mpi_info['size']}, Is Parallel: {mpi_info['is_parallel']}")