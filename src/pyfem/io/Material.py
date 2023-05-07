class Material:
    def __init__(self):
        self.name = None
        self.type = None
        self.data = None

    def to_string(self, level=1):
        BLUE = '\033[34m'
        END = '\033[0m'
        msg = BLUE + self.__str__() + END
        msg += '\n'
        for key, item in self.__dict__.items():
            msg += '  ' * level + f'|- {key}: {item}\n'
        return msg[:-1]


if __name__ == "__main__":
    material = Material()
    print(material.__dict__.keys())
    print(material)
    print(material.to_string())
