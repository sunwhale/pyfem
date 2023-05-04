class Dofs:
    def __init__(self):
        self.names = None
        self.family = None
        self.order = None

    def __str__(self):
        msg = ''
        for key, item in self.__dict__.items():
            msg += f'  |- {key}: {item}\n'
        return msg[5:-1]


if __name__ == "__main__":
    dofs = Dofs()
    print(dofs.__dict__.keys())
