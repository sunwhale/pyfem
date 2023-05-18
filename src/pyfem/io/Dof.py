from typing import Optional, List

from pyfem.utils.visualization import object_dict_to_string


class Dof:
    def __init__(self) -> None:
        self.names: List[str] = None  # type: ignore
        self.family: Optional[str] = None
        self.order: Optional[int] = None

    def to_string(self, level: int = 1) -> str:
        return object_dict_to_string(self, level)

    def show(self) -> None:
        print(self.to_string())


if __name__ == "__main__":
    dof = Dof()
    print(dof.__dict__.keys())
    print(dof)
    print(dof.to_string())
