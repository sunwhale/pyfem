from pyfem.utils.Constants import BLUE, END


class Solver:
    def __init__(self):
        self.type = None

    def to_string(self, level=1):
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
