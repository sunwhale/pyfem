# -*- coding: utf-8 -*-
"""

"""
from pyfem.utils.visualization import object_slots_to_string_ndarray


class Timer:
    """
    计时器类，用于存储求解过程中的时间信息。

    :ivar total_time: 总时间
    :vartype total_time: float

    :ivar time0: 上一个载荷步的时间
    :vartype time0: float

    :ivar time1: 当前载荷步的时间
    :vartype time1: float

    :ivar dtime: 当前载荷步的时间增量
    :vartype dtime: float

    :ivar increment: 当前增量步
    :vartype increment: int

    :ivar frame_ids: 帧列表
    :vartype frame_ids: list[int]
    """

    __slots_dict__: dict = {
        'total_time': ('float', '总时间'),
        'time0': ('float', '上一个载荷步的时间'),
        'time1': ('float', '当前载荷步的时间'),
        'dtime': ('float', '当前载荷步的时间增量'),
        'increment': ('int', '当前增量步'),
        'frame_ids': ('list[int]', '帧列表')
    }

    __slots__: list = [slot for slot in __slots_dict__.keys()]

    TOL_TIME: float = 1e-9

    def __init__(self) -> None:
        self.total_time: float = 1.0
        self.time0: float = 0.0
        self.time1: float = 0.0
        self.dtime: float = 1.0
        self.increment: int = 0
        self.frame_ids: list[int] = []

    def to_string(self, level: int = 1) -> str:
        return object_slots_to_string_ndarray(self, level)

    def show(self) -> None:
        print(self.to_string())

    def is_done(self) -> bool:
        if self.time1 * (1.0 + self.TOL_TIME) >= self.total_time:
            return True
        else:
            return False


if __name__ == "__main__":
    from pyfem.utils.visualization import print_slots_dict

    print_slots_dict(Timer.__slots_dict__)

    timer = Timer()
    timer.show()
