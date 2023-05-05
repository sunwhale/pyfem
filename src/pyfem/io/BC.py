class BC:
    def __init__(self):
        self.name = None
        self.type = None
        self.dofs = None
        self.boundary = None
        self.value = None

    def to_string(self):
        msg = '      |--- \n'
        for key, item in self.__dict__.items():
            msg += f'      |- {key}: {item}\n'
        return msg[:-1]


if __name__ == "__main__":
    bc = BC()
    print(bc.__dict__.keys())
