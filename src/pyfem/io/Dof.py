from typing import Optional, List

from pyfem.utils.colors import BLUE, END


class Dof:
    def __init__(self) -> None:
        self.names: List[str] = None  # type: ignore
        self.family: Optional[str] = None
        self.order: Optional[int] = None

    def to_string(self, level: int = 1) -> str:
        msg = BLUE + self.__str__() + END
        msg += '\n'
        for key, item in self.__dict__.items():
            msg += '  ' * level + f'|- {key}: {item}\n'
        return msg[:-1]


if __name__ == "__main__":
    dof = Dof()
    print(dof.__dict__.keys())
    print(dof)
    print(dof.to_string())
