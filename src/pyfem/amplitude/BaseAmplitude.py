# -*- coding: utf-8 -*-
"""

"""
from typing import Callable

from scipy.interpolate import interp1d  # type: ignore

from pyfem.utils.visualization import object_slots_to_string_ndarray


class BaseAmplitude:
    """
    幅值基类。

    :ivar start: 初值
    :vartype start: float

    :ivar f_amplitude: 幅值函数
    :vartype f_amplitude: Callable
    """

    __slots_dict__: dict = {
        'start': ('float', '初值'),
        'f_amplitude': ('Callable', '幅值函数')
    }

    __slots__: list = [slot for slot in __slots_dict__.keys()]

    def __init__(self) -> None:
        self.start: float = 0.0
        self.f_amplitude: Callable = interp1d([0, 1], [0, 1], kind='linear', fill_value='extrapolate')

    def to_string(self, level: int = 1) -> str:
        return object_slots_to_string_ndarray(self, level)

    def show(self) -> None:
        print(self.to_string())

    def get_amplitude(self, time: float) -> float:
        return self.f_amplitude(time) + self.start

    def set_f_amplitude(self, time: list[float], value: list[float]) -> None:
        self.f_amplitude = interp1d(time, value, kind='linear', fill_value='extrapolate')


if __name__ == "__main__":
    from pyfem.utils.visualization import print_slots_dict

    print_slots_dict(BaseAmplitude.__slots_dict__)

    amp = BaseAmplitude()
    amp.show()
