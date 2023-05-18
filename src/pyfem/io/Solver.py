from pyfem.utils.visualization import object_dict_to_string


class Solver:
    def __init__(self) -> None:
        self.type: str = None  # type: ignore

    def to_string(self, level: int = 1) -> str:
        return object_dict_to_string(self, level)

    def show(self) -> None:
        print(self.to_string())


if __name__ == "__main__":
    solver = Solver()
    print(solver.__dict__.keys())
    print(solver)
    print(solver.to_string())
