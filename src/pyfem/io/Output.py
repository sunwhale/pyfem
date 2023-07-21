# -*- coding: utf-8 -*-
"""

"""
from pyfem.io.BaseIO import BaseIO


class Output(BaseIO):
    """
    定义输出文件的格式和详细信息。

    :ivar type: 输出类型
    :vartype type: str

    :ivar field_outputs: 输出场变量列表
    :vartype field_outputs: list[str]

    :ivar on_screen: 是否在屏幕上显示
    :vartype on_screen: bool
    """

    __slots_dict__: dict = {
        'type': ('str', '输出类型'),
        'field_outputs': ('list[str]', '输出场变量列表'),
        'on_screen': ('bool', '是否在屏幕上显示')
    }

    __slots__: list = [slot for slot in __slots_dict__.keys()]

    def __init__(self) -> None:
        super().__init__()
        self.type: str = None  # type: ignore
        self.field_outputs: list[str] = None  # type: ignore
        self.on_screen: bool = None  # type: ignore


if __name__ == "__main__":
    from pyfem.utils.visualization import print_slots_dict

    print_slots_dict(Output.__slots_dict__)

    output = Output()
    output.show()
