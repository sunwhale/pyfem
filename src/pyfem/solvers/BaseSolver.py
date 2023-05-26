from numpy import ndarray, empty

from pyfem.assembly.Assembly import Assembly
from pyfem.io.Solver import Solver
from pyfem.utils.visualization import object_dict_to_string_ndarray


class BaseSolver:
    def __init__(self, assembly: Assembly, solver: Solver) -> None:
        self.assembly: Assembly = assembly
        self.solver: Solver = solver
        self.solution: ndarray = empty(0)

    def to_string(self, level: int = 1) -> str:
        return object_dict_to_string_ndarray(self, level)

    def show(self) -> None:
        print(self.to_string())

    def run(self) -> None:
        self.solve()
        self.update_field_variables()

    def solve(self) -> None:
        self.solution = empty(0)

    def update_field_variables(self) -> None:
        self.assembly.update_element_data(self.solution)
        self.assembly.update_field_variables()


if __name__ == "__main__":
    pass
