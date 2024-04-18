# -*- coding: utf-8 -*-
"""

"""
from typing import Union

from pyfem.assembly.Assembly import Assembly
from pyfem.io.Solver import Solver
from pyfem.solvers.BaseSolver import BaseSolver
from pyfem.solvers.LinearSolver import LinearSolver
from pyfem.solvers.NonlinearSolver import NonlinearSolver
from pyfem.solvers.TimeIntegrationNonlinearSolver import TimeIntegrationNonlinearSolver
from pyfem.solvers.ArcLengthSolver import ArcLengthSolver
from pyfem.utils.colors import error_style

SolverData = Union[BaseSolver, LinearSolver, NonlinearSolver, ArcLengthSolver]

solver_data_dict = {
    'LinearSolver': LinearSolver,
    'NonlinearSolver': NonlinearSolver,
    'ArcLengthSolver': ArcLengthSolver,
    'TimeIntegrationNonlinearSolver': TimeIntegrationNonlinearSolver,
}


def get_solver_data(assembly: Assembly, solver: Solver) -> SolverData:
    """
    工厂函数，用于根据求解器属性生产不同的求解器对象。

    Args:
        assembly(Assembly): 装配体对象
        solver(Solver): 求解器属性

    :return: 求解器对象
    :rtype: SolverData
    """

    class_name = f'{solver.type}'.strip().replace(' ', '')

    if class_name in solver_data_dict:
        return solver_data_dict[class_name](assembly=assembly,
                                            solver=solver)
    else:
        error_msg = f'{class_name} solver is not supported.\n'
        error_msg += f'The allowed solver types are {list(solver_data_dict.keys())}.'
        raise NotImplementedError(error_style(error_msg))


if __name__ == "__main__":
    pass
