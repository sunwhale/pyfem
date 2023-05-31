# -*- coding: utf-8 -*-
"""

"""
from pyfem.utils.visualization import object_dict_to_string


class Timer:
    def __init__(self) -> None:
        self.total_time: float = 0.0
        self.time0: float = 0.0
        self.time1: float = 0.0
        self.dtime: float = 0.0

    def to_string(self, level: int = 1) -> str:
        return object_dict_to_string(self, level)

    def show(self) -> None:
        print(self.to_string())

    def is_done(self) -> bool:
        if self.time1 * (1 + 1e-6) >= self.total_time:
            return True
        else:
            return False


if __name__ == "__main__":
    timer = Timer()
    timer.show()
