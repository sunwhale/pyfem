from typing import Optional

from pyfem.utils.visualization import object_dict_to_string


class Output:
    def __init__(self) -> None:
        self.type: Optional[str] = None
        self.on_screen: Optional[bool] = None

    def to_string(self, level: int = 1) -> str:
        return object_dict_to_string(self, level)

    def show(self) -> None:
        print(self.to_string())


if __name__ == "__main__":
    output = Output()
    print(output.__dict__.keys())
    print(output)
    print(output.to_string())
