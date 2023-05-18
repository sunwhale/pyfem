from typing import Optional

from pyfem.utils.colors import BLUE, END


class Output:
    def __init__(self) -> None:
        self.type: Optional[str] = None
        self.on_screen: Optional[bool] = None

    def to_string(self, level: int = 1) -> str:
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
