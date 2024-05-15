# -*- coding: utf-8 -*-
"""

"""
from pyfem.io.BaseIO import BaseIO


class Output(BaseIO):
    """
    定义输出文件的格式和详细信息。

    :ivar name: 输出名称
    :vartype name: str

    :ivar type: 输出类型
    :vartype type: str

    :ivar field_outputs: 输出场变量列表
    :vartype field_outputs: list[str]

    :ivar is_save: 是否保存结果文件
    :vartype is_save: bool
    """

    __slots_dict__: dict = {
        'name': ('str', '输出名称'),
        'type': ('str', '输出类型'),
        'field_outputs': ('list[str]', '输出场变量列表'),
        'is_save': ('bool', '是否保存结果文件')
    }

    __slots__: list = [slot for slot in __slots_dict__.keys()]

    allowed_types: list = ['hdf5', 'vtk']

    def __init__(self) -> None:
        super().__init__()
        self.name: str = None  # type: ignore
        self.type: str = None  # type: ignore
        self.field_outputs: list[str] = None  # type: ignore
        self.is_save: bool = None  # type: ignore


if __name__ == "__main__":
    from pyfem.utils.visualization import print_slots_dict

    print_slots_dict(Output.__slots_dict__)

    output = Output()
    output.show()
