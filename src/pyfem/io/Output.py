from pyfem.utils.Constants import BLUE, END


class Output:
    def __init__(self):
        self.type = None
        self.on_screen = None

    def to_string(self, level=1):
        msg = BLUE + self.__str__() + END
        msg += '\n'
        for key, item in self.__dict__.items():
            msg += '  ' * level + f'|- {key}: {item}\n'
        return msg[:-1]


if __name__ == "__main__":
    output = Output()
    print(output.__dict__.keys())
    print(output)
    print(output.to_string())
