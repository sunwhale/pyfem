class Domain:
    def __init__(self):
        self.name = None
        self.type = None
        self.material_name = None

    def to_string(self, level=1):
        BLUE = '\033[34m'
        END = '\033[0m'
        msg = BLUE + self.__str__() + END
        msg += '\n'
        for key, item in self.__dict__.items():
            msg += '  ' * level + f'|- {key}: {item}\n'
        return msg[:-1]


if __name__ == "__main__":
    domain = Domain()
    print(domain.__dict__.keys())
    print(domain)
    print(domain.to_string())