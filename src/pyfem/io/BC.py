from typing import Optional, List

from pyfem.utils.visualization import object_dict_to_string


class BC:
    def __init__(self) -> None:
        self.name: Optional[str] = None
        self.type: Optional[str] = None
        self.dof: Optional[List[str]] = None
        self.node_sets: Optional[List[str]] = None
        self.element_sets: Optional[List[str]] = None
        self.value: Optional[float] = None

    def to_string(self, level: int = 1) -> str:
        return object_dict_to_string(self, level)

    def show(self) -> None:
        print(self.to_string())


if __name__ == "__main__":
    bc = BC()
    bc.show()
