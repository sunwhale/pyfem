from pyfem.assembly.Assembly import Assembly
from pyfem.io.Solver import Solver
from pyfem.solvers.BaseSolver import BaseSolver
from pyfem.solvers.LinearSolver import LinearSolver
from pyfem.solvers.NonlinearSolver import NonlinearSolver
from pyfem.utils.colors import error_style

solver_data_dict = {
    'LinearSolver': LinearSolver,
    'NonlinearSolver': NonlinearSolver
}


def get_solver_data(assembly: Assembly, solver: Solver) -> BaseSolver:
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
