# -*- coding: utf-8 -*-
"""

"""
from typing import List, Tuple, Dict

from pyfem.utils.colors import error_style
from pyfem.io.BaseIO import BaseIO


class Material(BaseIO):
    """
    Material类用于存储配置文件中定义的材料属性。

    当 self.is_read_only = True 时：
        1. Material 类的所有属性在首次被赋非None值后不能再被修改和删除，
        2. 此时许可的属性关键字存储在self.slots中。
    """
    __slots__: Tuple = ('name',
                        'category',
                        'type',
                        'data')

    allowed_categories_types: Dict = {
        None: [None],
        'Elastic': ['Isotropic'],
        'Plastic': ['KinematicHardening'],
        'ViscoElastic': ['Maxwell'],
        'Thermal': ['Isotropic'],
        'PhaseField': ['Damage'],
        'MechanicalThermal': ['Expansion']
    }

    allowed_keys_values: Dict = {
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
        self.data: List[float] = None  # type: ignore

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
    material = Material()
    material.show()
