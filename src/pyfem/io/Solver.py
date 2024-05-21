# -*- coding: utf-8 -*-
"""

"""
from pyfem.io.BaseIO import BaseIO


class Solver(BaseIO):
    """
    定义求解器属性。

    :ivar type: 求解器类型
    :vartype type: str

    :ivar option: 求解器选项
    :vartype option: str

    :ivar total_time: 总时间
    :vartype total_time: float

    :ivar start_time: 开始时间
    :vartype start_time: float

    :ivar max_increment: 最大增量步数量
    :vartype max_increment: int

    :ivar initial_dtime: 初始时间增量步长
    :vartype initial_dtime: float

    :ivar max_dtime: 最大时间增量步长
    :vartype max_dtime: float

    :ivar min_dtime: 最小时间增量步长
    :vartype min_dtime: float
    """

    __slots_dict__: dict = {
        'type': ('str', '求解器类型'),
        'option': ('str', '求解器选项'),
        'total_time': ('float', '总时间'),
        'start_time': ('float', '开始时间'),
        'max_increment': ('int', '最大增量步数量'),
        'initial_dtime': ('float', '初始时间增量步长'),
        'max_dtime': ('float', '最大时间增量步长'),
        'min_dtime': ('float', '最小时间增量步长')
    }

    __slots__: list = [slot for slot in __slots_dict__.keys()]

    allowed_types_options: dict = {
        'NonlinearSolver': ['NewtonRaphson', 'InitialTangent'],
        'LinearSolver': [''],
    }

    def __init__(self) -> None:
        super().__init__()
        self.type: str = None  # type: ignore
        self.option: str = None  # type: ignore
        self.total_time: float = None  # type: ignore
        self.start_time: float = None  # type: ignore
        self.max_increment: int = None  # type: ignore
        self.initial_dtime: float = None  # type: ignore
        self.max_dtime: float = None  # type: ignore
        self.min_dtime: float = None  # type: ignore


if __name__ == "__main__":
    from pyfem.utils.visualization import print_slots_dict

    print_slots_dict(Solver.__slots_dict__)

    solver = Solver()
    solver.show()
