# -*- coding: utf-8 -*-
"""

"""
from typing import List, Callable

from scipy.interpolate import interp1d  # type: ignore

from pyfem.utils.visualization import object_slots_to_string_ndarray


class BaseAmplitude:
    __slots__ = ('start',
                 'f_amplitude')

    def __init__(self) -> None:
        self.start: float = 0.0
        self.f_amplitude: Callable = interp1d([0, 1], [0, 1], kind='linear', fill_value='extrapolate')

    def to_string(self, level: int = 1) -> str:
        return object_slots_to_string_ndarray(self, level)

    def show(self) -> None:
        print(self.to_string())

    def get_amplitude(self, time: float) -> float:
        return self.f_amplitude(time) + self.start

    def set_f_amplitude(self, time: List[float], value: List[float]) -> None:
        self.f_amplitude = interp1d(time, value, kind='linear', fill_value='extrapolate')


if __name__ == "__main__":
    amp = BaseAmplitude()
    amp.show()
