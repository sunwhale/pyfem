from typing import Optional, List

from pyfem.utils.colors import BLUE, END


class Section:
    def __init__(self) -> None:
        self.name: str = None  # type: ignore
        self.category: str = None  # type: ignore
        self.type: str = None  # type: ignore
        self.option: Optional[str] = None
        self.element_sets: List[str] = None  # type: ignore
        self.material_name: str = None  # type: ignore
        self.data: List = None  # type: ignore

    def to_string(self, level: int = 1) -> str:
        msg = BLUE + self.__str__() + END
        msg += '\n'
        for key, item in self.__dict__.items():
            msg += '  ' * level + f'|- {key}: {item}\n'
        return msg[:-1]


if __name__ == "__main__":
    section = Section()
    print(section.__dict__.keys())
    print(section)
    print(section.to_string())
