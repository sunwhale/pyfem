# -*- coding: utf-8 -*-
"""

"""
from pyfem.utils.visualization import object_dict_to_string_ndarray


class BaseAmplitude:
    def __init__(self) -> None:
        self.start: float = 0.0

    def to_string(self, level: int = 1) -> str:
        return object_dict_to_string_ndarray(self, level)

    def show(self) -> None:
        print(self.to_string())

    def get_amplitude(self, time: float) -> float:
        return 1.0


if __name__ == "__main__":
    amp = BaseAmplitude()
    amp.show()
