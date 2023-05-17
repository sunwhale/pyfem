from pyfem.utils.colors import BLUE, END


class BC:
    def __init__(self):
        self.name = None
        self.type = None
        self.dof = None
        self.boundary = None
        self.value = None

    def to_string(self, level=1):
        msg = BLUE + self.__str__() + END
        msg += '\n'
        for key, item in self.__dict__.items():
            msg += '  ' * level + f'|- {key}: {item}\n'
        return msg[:-1]


if __name__ == "__main__":
    bc = BC()
    print(bc.__dict__.keys())
    print(bc)
    print(bc.to_string())
