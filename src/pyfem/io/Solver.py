from pyfem.utils.colors import BLUE, END


class Solver:
    def __init__(self) -> None:
        self.type: str = None  # type: ignore

    def to_string(self, level: int = 1) -> str:
        msg = BLUE + self.__str__() + END
        msg += '\n'
        for key, item in self.__dict__.items():
            msg += '  ' * level + f'|- {key}: {item}\n'
        return msg[:-1]


if __name__ == "__main__":
    solver = Solver()
    print(solver.__dict__.keys())
    print(solver)
    print(solver.to_string())
