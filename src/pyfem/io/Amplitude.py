# -*- coding: utf-8 -*-
"""

"""
from pyfem.io.BaseIO import BaseIO


class Amplitude(BaseIO):
    """
    定义幅值。

    :ivar name: 幅值名称
    :vartype name: str

    :ivar type: 幅值类型
    :vartype type: str

    :ivar start: 幅值起始点
    :vartype start: float

    :ivar data: 幅值数据列表
    :vartype data: list[float]
    """

    __slots_dict__: dict = {
        'name': ('str', '幅值名称'),
        'type': ('str', '幅值类型'),
        'start': ('float', '幅值起始点'),
        'data': ('list[float]', '幅值数据列表')
    }

    __slots__: list = [slot for slot in __slots_dict__.keys()]

    def __init__(self) -> None:
        super().__init__()
        self.name: str = None  # type: ignore
        self.type: str = None  # type: ignore
        self.start: float = None  # type: ignore
        self.data: list[float] = None  # type: ignore


if __name__ == "__main__":
    from pyfem.utils.visualization import print_slots_dict

    print_slots_dict(Amplitude.__slots_dict__)

    amp = Amplitude()
    amp.show()
