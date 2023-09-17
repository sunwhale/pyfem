# -*- coding: utf-8 -*-
"""

"""
from pyfem.io.BaseIO import BaseIO
from pyfem.utils.colors import error_style


class Material(BaseIO):
    """
    定义材料属性。

    :ivar name: 材料名称
    :vartype name: str

    :ivar category: 材料类别
    :vartype category: str

    :ivar type: 材料类型
    :vartype type: str

    :ivar data: 材料数据列表
    :vartype data: list[float]

    :ivar data_dict: 材料数据字典
    :vartype data_dict: dict[str, any]
    """

    __slots_dict__: dict = {
        'name': ('str', '材料名称'),
        'category': ('str', '材料类别'),
        'type': ('str', '材料类型'),
        'data': ('list[float]', '材料数据列表'),
        'data_dict': ('dict[str, any]', '材料数据字典')
    }

    __slots__: list = [slot for slot in __slots_dict__.keys()]

    allowed_categories_types: dict = {
        None: [None],
        'Elastic': ['Isotropic'],
        'Plastic': ['KinematicHardening', 'Crystal', 'CrystalGNDs'],
        'ViscoElastic': ['Maxwell'],
        'Thermal': ['Isotropic'],
        'PhaseField': ['Damage'],
        'MechanicalThermal': ['Expansion']
    }

    allowed_keys_values: dict = {
        'category': allowed_categories_types.keys(),
        'type': []
    }

    for types in allowed_categories_types.values():
        allowed_keys_values['type'] += types

    def __init__(self) -> None:
        super().__init__()
        self.name: str = None  # type: ignore
        self.category: str = None  # type: ignore
        self.type: str = None  # type: ignore
        self.data: list[float] = None  # type: ignore
        self.data_dict: dict = None  # type: ignore

    def __setattr__(self, key, value) -> None:
        if self.is_read_only:
            if key not in self.__slots__:
                error_msg = f'{key} is not an allowable attribute keyword of {type(self).__name__}'
                raise AttributeError(error_style(error_msg))

            elif hasattr(self, key) and self.__getattribute__(key) is not None:
                error_msg = f'attribute {type(self).__name__}.{key} is READ ONLY'
                raise PermissionError(error_style(error_msg))

            else:
                for allowed_key, allowed_values in self.allowed_keys_values.items():
                    if key == allowed_key and value not in allowed_values:
                        error_msg = f'{value} is unsupported for {type(self).__name__}.{key}\n'
                        error_msg += f'The allowed values of {type(self).__name__}.{key} are {self.allowed_keys_values[key]}'
                        raise AttributeError(error_style(error_msg))

                super().__setattr__(key, value)
        else:
            super().__setattr__(key, value)


if __name__ == "__main__":
    from pyfem.utils.visualization import print_slots_dict

    print_slots_dict(Material.__slots_dict__)

    material = Material()
    material.show()
