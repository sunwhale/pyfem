# -*- coding: utf-8 -*-
"""

"""
from pyfem.io.BaseIO import BaseIO


class BC(BaseIO):
    """
    定义边界条件。

    :ivar name: 边界条件名称
    :vartype name: str

    :ivar category: 边界条件类别
    :vartype category: str

    :ivar type: 边界条件类型
    :vartype type: str

    :ivar dof: 自由度列表
    :vartype dof: list[str]

    :ivar node_sets: 节点集合列表
    :vartype node_sets: list[str]

    :ivar element_sets: 单元集合列表
    :vartype element_sets: list[str]

    :ivar bc_element_sets: 边界单元集合列表
    :vartype bc_element_sets: list[str]

    :ivar value: 边界条件数值
    :vartype value: float

    :ivar amplitude_name: 边界条件幅值名称
    :vartype amplitude_name: str
    """

    __slots_dict__: dict = {
        'name': ('str', '边界条件名称'),
        'category': ('str', '边界条件类别'),
        'type': ('str', '边界条件类型'),
        'dof': ('list[str]', '自由度列表'),
        'node_sets': ('list[str]', '节点集合列表'),
        'element_sets': ('list[str]', '单元集合列表'),
        'bc_element_sets': ('list[str]', '边界单元集合列表'),
        'value': ('float', '边界条件数值'),
        'amplitude_name': ('str', '边界条件幅值名称')
    }

    __slots__: list = [slot for slot in __slots_dict__.keys()]

    def __init__(self) -> None:
        super().__init__()
        self.name: str = None  # type: ignore
        self.category: str = None  # type: ignore
        self.type: str = None  # type: ignore
        self.dof: list[str] = None  # type: ignore
        self.node_sets: list[str] = None  # type: ignore
        self.element_sets: list[str] = None  # type: ignore
        self.bc_element_sets: list[str] = None  # type: ignore
        self.value: float = None  # type: ignore
        self.amplitude_name: str = None  # type: ignore


if __name__ == "__main__":
    from pyfem.utils.visualization import print_slots_dict

    print_slots_dict(BC.__slots_dict__)

    bc = BC()
    bc.show()
