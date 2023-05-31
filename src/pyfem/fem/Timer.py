# -*- coding: utf-8 -*-
"""

"""
from typing import List

from pyfem.utils.visualization import object_dict_to_string


class Timer:
    """
    计时器类，用于存储求解过程中的时间信息。
    """
    TOL_TIME = 1e-6

    def __init__(self) -> None:
        self.total_time: float = 0.0
        self.time0: float = 0.0
        self.time1: float = 0.0
        self.dtime: float = 0.0
        self.increment: int = 0
        self.frame_ids: List[int] = []

    def to_string(self, level: int = 1) -> str:
        return object_dict_to_string(self, level)

    def show(self) -> None:
        print(self.to_string())

    def is_done(self) -> bool:
        if self.time1 * (1.0 + self.TOL_TIME) >= self.total_time:
            return True
        else:
            return False


if __name__ == "__main__":
    timer = Timer()
    timer.show()
