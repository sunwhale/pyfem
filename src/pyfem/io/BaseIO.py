# -*- coding: utf-8 -*-
"""

"""
from typing import Dict, List

from pyfem.utils.colors import error_style
from pyfem.utils.visualization import object_slots_to_string


class BaseIO:
    """
    读取属性配置的基类。

    当 self.is_read_only = True 时：BaseIO子类的所有属性在第一次被赋予非None值后变为只读状态，不能被修改或删除。

    """
    __slots__: List = []

    is_read_only: bool = True

    def __init__(self) -> None:
        pass

    def __setattr__(self, key, value):
        if self.is_read_only:
            if key not in self.__slots__:
                error_msg = f'\'{key}\' is not an allowable attribute keyword of \'{type(self).__name__}\''
                raise AttributeError(error_style(error_msg))
            elif hasattr(self, key) and self.__getattribute__(key) is not None:
                error_msg = f'attribute \'{type(self).__name__}.{key}\' is READ ONLY'
                raise PermissionError(error_style(error_msg))
            else:
                super().__setattr__(key, value)
        else:
            super().__setattr__(key, value)

    def __delattr__(self, key):
        if self.is_read_only:
            error_msg = f'attribute \'{type(self).__name__}.{key}\' is READ ONLY'
            raise PermissionError(error_style(error_msg))
        else:
            super().__delattr__(key)

    def to_string(self, level: int = 1) -> str:
        return object_slots_to_string(self, level)

    def show(self) -> None:
        print(self.to_string())

    def set_io_values(self, io_dict: Dict) -> None:
        for key, item in io_dict.items():
            self.__setattr__(key, item)


if __name__ == "__main__":
    io = BaseIO()
    io.show()
