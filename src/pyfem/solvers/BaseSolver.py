from numpy import ndarray, empty

from pyfem.assembly.Assembly import Assembly
from pyfem.io.Solver import Solver
from pyfem.utils.visualization import object_dict_to_string_ndarray


class BaseSolver:
    def __init__(self) -> None:
        self.assembly: Assembly = None  # type: ignore
        self.solver: Solver = None  # type: ignore
        self.solution: ndarray = empty(0)

    def to_string(self, level: int = 1) -> str:
        return object_dict_to_string_ndarray(self, level)

    def show(self) -> None:
        print(self.to_string())

    def run(self) -> None:
        pass

    def solve(self) -> None:
        pass


if __name__ == "__main__":
    pass
