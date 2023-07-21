# -*- coding: utf-8 -*-
"""

"""
from pyfem.io.BaseIO import BaseIO


class Dof(BaseIO):
    """
    定义求解自由度。

    :ivar names: 自由度名称列表
    :vartype names: list[str]

    :ivar family: 自由度名称类型
    :vartype family: str

    :ivar order: 自由度阶次
    :vartype order: int
    """

    __slots_dict__: dict = {
        'names': ('list[str]', '自由度名称列表'),
        'family': ('str', '自由度名称类型'),
        'order': ('int', '自由度阶次')
    }

    __slots__: list = [slot for slot in __slots_dict__.keys()]

    def __init__(self) -> None:
        super().__init__()
        self.names: list[str] = None  # type: ignore
        self.family: str = None  # type: ignore
        self.order: int = None  # type: ignore


if __name__ == "__main__":
    from pyfem.utils.visualization import print_slots_dict

    print_slots_dict(Dof.__slots_dict__)

    dof = Dof()
    dof.show()
