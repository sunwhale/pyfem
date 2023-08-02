# -*- coding: utf-8 -*-
"""

"""
from numpy import ndarray, empty

from pyfem.fem.Timer import Timer
from pyfem.io.Material import Material
from pyfem.io.Section import Section
from pyfem.utils.visualization import object_slots_to_string_ndarray


class BaseMaterial:
    """
    材料对象的基类。

    :ivar material: 材料属性
    :vartype material: Material

    :ivar dimension: 空间维度
    :vartype dimension: int

    :ivar section: 截面属性
    :vartype section: Section

    :ivar allowed_section_types: 当前材料许可的截面类型
    :vartype allowed_section_types: tuple

    :ivar tangent: 切线刚度矩阵
    :vartype tangent: ndarray

    :ivar output: 输出变量字典
    :vartype output: dict[str, ndarray]

    :ivar data_keys: 材料属性数据关键字列表
    :vartype data_keys: list[str]

    :ivar data_dict: 材料属性数据字典
    :vartype data_dict: dict[str, float]
    """

    __slots_dict__: dict = {
        'material': ('Material', '材料属性'),
        'dimension': ('int', '空间维度'),
        'section': ('Section', '截面属性'),
        'allowed_section_types': ('tuple', '当前材料许可的截面类型'),
        'tangent': ('ndarray', '切线刚度矩阵'),
        'output': ('dict[str, ndarray]', '输出变量字典'),
        'data_keys': ('list[str]', '材料属性数据关键字列表'),
        'data_dict': ('dict[str, float]', '材料属性数据字典')
    }

    __slots__: list = [slot for slot in __slots_dict__.keys()]

    def __init__(self, material: Material, dimension: int, section: Section) -> None:
        self.material: Material = material
        self.dimension: int = dimension
        self.section: Section = section
        self.allowed_section_types: tuple = ()
        self.tangent: ndarray = empty(0)
        self.output: dict[str, ndarray] = {}
        self.data_keys: list[str] = []
        self.data_dict: dict[str, float] = {}

    def get_section_type_error_msg(self) -> str:
        return f'\'{self.section.type}\' is not the allowed section types {self.allowed_section_types} of the material \'{self.material.name}\' -> {type(self).__name__}, please check the definition of the section \'{self.section.name}\''

    def get_data_length_error_msg(self) -> str:
        return f'the length of \'data\' -> {self.material.data} of \'{self.material.name}\' -> {type(self).__name__} must be {len(self.data_keys)} and stored in the order of {self.data_keys}'

    def to_string(self, level: int = 1) -> str:
        return object_slots_to_string_ndarray(self, level)

    def show(self) -> None:
        print(self.to_string())

    def create_tangent(self) -> None:
        pass

    def get_tangent(self, variable: dict[str, ndarray],
                    state_variable: dict[str, ndarray],
                    state_variable_new: dict[str, ndarray],
                    element_id: int,
                    iqp: int,
                    ntens: int,
                    ndi: int,
                    nshr: int,
                    timer: Timer) -> tuple[ndarray, dict[str, ndarray]]:
        return self.tangent, self.output


if __name__ == "__main__":
    from pyfem.utils.visualization import print_slots_dict

    print_slots_dict(BaseMaterial.__slots_dict__)
