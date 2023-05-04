class Material:
    def __init__(self):
        self.name = None
        self.type = None
        self.data = None

    def to_string(self):
        msg = ''
        for key, item in self.__dict__.items():
            msg += f'      |- {key}: {item}\n'
        return msg[:-1]


if __name__ == "__main__":
    material = Material()
    print(material.__dict__.keys())
