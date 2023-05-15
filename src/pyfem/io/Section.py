from pyfem.utils.colors import BLUE, END


class Section:
    def __init__(self):
        self.name = None
        self.category = None
        self.type = None
        self.element_set_name = None
        self.material_name = None
        self.data = None

    def to_string(self, level=1):
        msg = BLUE + self.__str__() + END
        msg += '\n'
        for key, item in self.__dict__.items():
            msg += '  ' * level + f'|- {key}: {item}\n'
        return msg[:-1]


if __name__ == "__main__":
    section = Section()
    print(section.__dict__.keys())
    print(section)
    print(section.to_string())
