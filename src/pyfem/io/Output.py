from pyfem.utils.visualization import object_dict_to_string


class Output:
    def __init__(self) -> None:
        self.type: str = None  # type: ignore
        self.field_outputs: str = None  # type: ignore
        self.on_screen: bool = None  # type: ignore

    def to_string(self, level: int = 1) -> str:
        return object_dict_to_string(self, level)

    def show(self) -> None:
        print(self.to_string())


if __name__ == "__main__":
    output = Output()
    print(output.__dict__.keys())
    print(output)
    print(output.to_string())
