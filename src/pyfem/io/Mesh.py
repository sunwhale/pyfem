# -*- coding: utf-8 -*-
"""

"""
from pyfem.io.BaseIO import BaseIO


class Mesh(BaseIO):
    """
    定义需要读取的网格信息。

    :ivar type: 网格类型
    :vartype type: str

    :ivar file: 网格文件路径
    :vartype file: str
    """

    __slots_dict__: dict = {
        'type': ('str', '网格类型'),
        'file': ('str', '网格文件路径')
    }

    __slots__: list = [slot for slot in __slots_dict__.keys()]

    def __init__(self) -> None:
        super().__init__()
        self.type: str = None  # type: ignore
        self.file: str = None  # type: ignore


if __name__ == "__main__":
    from pyfem.utils.visualization import print_slots_dict

    print_slots_dict(Mesh.__slots_dict__)

    mesh = Mesh()
    mesh.show()
