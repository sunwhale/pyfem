from typing import List

from pyfem.utils.visualization import object_dict_to_string


class BC:
    def __init__(self) -> None:
        self.name: str = None  # type: ignore
        self.type: str = None  # type: ignore
        self.dof: List[str] = None  # type: ignore
        self.node_sets: List[str] = None  # type: ignore
        self.element_sets: List[str] = None  # type: ignore
        self.value: float = None  # type: ignore

    def to_string(self, level: int = 1) -> str:
        return object_dict_to_string(self, level)

    def show(self) -> None:
        print(self.to_string())


if __name__ == "__main__":
    bc = BC()
    bc.show()
