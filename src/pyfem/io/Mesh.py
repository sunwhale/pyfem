class Mesh:
    def __init__(self):
        self.type = None
        self.file = None

    def __str__(self):
        msg = ''
        for key, item in self.__dict__.items():
            msg += f'  |- {key}: {item}\n'
        return msg[5:-1]


if __name__ == "__main__":
    mesh = Mesh()
    print(mesh.__dict__.keys())
