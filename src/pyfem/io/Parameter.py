# -*- coding: utf-8 -*-
"""

"""
from pyfem.io.BaseIO import BaseIO


class Parameter(BaseIO):
    """
    定义需要读取的参数信息。

    :ivar file: 参数文件路径
    :vartype file: str
    """

    __slots_dict__: dict = {
        'file': ('str', '参数文件路径')
    }

    __slots__: list = [slot for slot in __slots_dict__.keys()]

    def __init__(self) -> None:
        super().__init__()
        self.file: str = None  # type: ignore


if __name__ == "__main__":
    from pyfem.utils.visualization import print_slots_dict

    print_slots_dict(Parameter.__slots_dict__)

    mesh = Parameter()
    mesh.show()
