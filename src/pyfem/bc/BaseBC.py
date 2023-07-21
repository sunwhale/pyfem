# -*- coding: utf-8 -*-
"""

"""
from typing import Optional, Callable

from numpy import empty, ndarray

from pyfem.amplitude.BaseAmplitude import BaseAmplitude
from pyfem.amplitude.get_amplitude_data import get_amplitude_data
from pyfem.io.Amplitude import Amplitude
from pyfem.io.BC import BC
from pyfem.io.Dof import Dof
from pyfem.io.Solver import Solver
from pyfem.mesh.MeshData import MeshData
from pyfem.utils.visualization import object_slots_to_string_ndarray


class BaseBC:
    """
    边界条件对象基类。

    :ivar name: 边界条件名称
    :vartype name: str

    :ivar category: 边界条件类别
    :vartype category: str

    :ivar type: 边界条件类型
    :vartype type: str

    :ivar dof: 自由度列表
    :vartype dof: list[str]

    :ivar node_sets: 节点集合列表
    :vartype node_sets: list[str]

    :ivar element_sets: 单元集合列表
    :vartype element_sets: list[str]

    :ivar bc_element_sets: 边界单元集合列表
    :vartype bc_element_sets: list[str]

    :ivar value: 边界条件数值
    :vartype value: float

    :ivar amplitude_name: 边界条件幅值名称
    :vartype amplitude_name: str
    """

    __slots_dict__: dict = {
        'bc': ('BC', '边界条件属性'),
        'dof': ('Dof', '自由度属性'),
        'mesh_data': ('MeshData', '幅值起始点'),
        'solver': ('Solver', '求解器属性'),
        'amplitude': ('Optional[Amplitude]', '幅值属性'),
        'amplitude_data': ('BaseAmplitude', '幅值对象'),
        'get_amplitude': ('Callable', '获取给定数值所对应的幅值'),
        'bc_node_ids': ('ndarray', '边界节点编号列表'),
        'bc_element_ids': ('ndarray', '边界单元编号列表'),
        'dof_ids': ('ndarray', '自由度编号列表'),
        'dof_values': ('ndarray', '自由度数值列表'),
        'bc_fext': ('ndarray', '等效节点力列表'),
        'bc_surface': ('list[tuple[int, str]]', '边界表面')
    }

    __slots__: list = [slot for slot in __slots_dict__.keys()]

    def __init__(self, bc: BC, dof: Dof, mesh_data: MeshData, solver: Solver, amplitude: Optional[Amplitude]) -> None:
        self.bc: BC = bc
        self.dof: Dof = dof
        self.mesh_data: MeshData = mesh_data
        self.solver: Solver = solver
        self.amplitude: Optional[Amplitude] = amplitude
        if self.amplitude is not None:
            self.amplitude_data: BaseAmplitude = get_amplitude_data(self.amplitude)
        else:
            self.amplitude_data = BaseAmplitude()
            self.amplitude_data.set_f_amplitude([0, solver.total_time], [0, 1])
        self.get_amplitude: Callable = self.amplitude_data.get_amplitude
        self.bc_node_ids: ndarray = empty(0)
        self.bc_element_ids: ndarray = empty(0)
        self.dof_ids: ndarray = empty(0)
        self.dof_values: ndarray = empty(0)
        self.bc_fext: ndarray = empty(0)
        self.bc_surface: list[tuple[int, str]] = []

    def to_string(self, level: int = 1) -> str:
        return object_slots_to_string_ndarray(self, level)

    def show(self) -> None:
        print(self.to_string())


if __name__ == "__main__":
    from pyfem.utils.visualization import print_slots_dict

    print_slots_dict(BC.__slots_dict__)
