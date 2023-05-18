from typing import Optional, List

from pyfem.utils.colors import BLUE, END


class BC:
    def __init__(self) -> None:
        self.name: Optional[str] = None
        self.type: Optional[str] = None
        self.dof: Optional[List[str]] = None
        self.node_sets: Optional[List[str]] = None
        self.value: Optional[float] = None

    def to_string(self, level: int = 1) -> str:
        msg = BLUE + self.__str__() + END
        msg += '\n'
        for key, item in self.__dict__.items():
            msg += '  ' * level + f'|- {key}: {item}\n'
        return msg[:-1]


if __name__ == "__main__":
    bc = BC()
    print(bc.__dict__.keys())
    print(bc)
    print(bc.to_string())
