# -*- coding: utf-8 -*-
"""

"""

from pyfem.io.BaseIO import BaseIO


class Section(BaseIO):
    """
    定义截面属性。

    :ivar name: 截面名称
    :vartype name: str

    :ivar category: 截面类别
    :vartype category: str

    :ivar type: 截面类型
    :vartype type: str

    :ivar option: 截面选项
    :vartype option: str

    :ivar element_sets: 单元集合列表，一个截面可以包含多个单元集合
    :vartype element_sets: list[str]

    :ivar material_names: 材料列表，一个截面可以包含多个材料属性，例如热传导系数+弹性模量
    :vartype material_names: list[str]

    :ivar data: 截面数据
    :vartype data: list[float]

    :ivar data_dict: 截面数据字典
    :vartype data_dict: dict[str, any]
    """

    __slots_dict__: dict = {
        'name': ('str', '截面名称'),
        'category': ('str', '截面类别'),
        'type': ('str', '截面类型'),
        'option': ('str', '截面选项'),
        'element_sets': ('list[str]', '单元集合列表，一个截面可以包含多个单元集合'),
        'material_names': ('list[str]', '材料列表，一个截面可以包含多个材料属性，例如热传导系数+弹性模量'),
        'data': ('list[float]', '截面数据'),
        'data_dict': ('dict[str, any]', '截面数据字典')
    }

    __slots__: list = [slot for slot in __slots_dict__.keys()]

    def __init__(self) -> None:
        super().__init__()
        self.name: str = None  # type: ignore
        self.category: str = None  # type: ignore
        self.type: str = None  # type: ignore
        self.option: str = None  # type: ignore
        self.element_sets: list[str] = None  # type: ignore
        self.material_names: list[str] = None  # type: ignore
        self.data: list[float] = None  # type: ignore
        self.data_dict: dict[str, any] = None  # type: ignore


if __name__ == "__main__":
    from pyfem.utils.visualization import print_slots_dict

    print_slots_dict(Section.__slots_dict__)

    section = Section()
    section.show()
