# -*- coding: utf-8 -*-
"""

"""
from typing import Optional, Callable

import numpy as np

from pyfem.amplitude.BaseAmplitude import BaseAmplitude
from pyfem.amplitude.get_amplitude_data import get_amplitude_data, AmplitudeData
from pyfem.io.Amplitude import Amplitude
from pyfem.io.BC import BC
from pyfem.io.Dof import Dof
from pyfem.io.Solver import Solver
from pyfem.isoelements.IsoElementShape import iso_element_shape_dict
from pyfem.isoelements.get_iso_element_type import get_iso_element_type
from pyfem.mesh.MeshData import MeshData
from pyfem.utils.colors import error_style
from pyfem.utils.visualization import object_slots_to_string_ndarray


class BaseBC:
    r"""
    **边界条件对象基类**

    其子类将基于边界条件的属性、自由度属性、网格对象、求解器属性和幅值属性获取系统线性方程组 :math:`{\mathbf{K u}} = {\mathbf{f}}` 中对应自由度 :math:`{\mathbf{u}}` 或等式右边项 :math:`{\mathbf{f}}` 的约束信息。

    当前支持的边界条件类型为Dirichlet边界和Neumann边界（Robin边界暂不支持）。

    :ivar bc: 边界条件属性
    :vartype bc: BC

    :ivar dof: 自由度属性
    :vartype dof: Dof

    :ivar mesh_data: 网格对象
    :vartype mesh_data: MeshData

    :ivar solver: 求解器属性
    :vartype solver: Solver

    :ivar amplitude: 幅值属性
    :vartype amplitude: Optional[Amplitude]

    :ivar amplitude_data: 幅值对象
    :vartype amplitude_data: BaseAmplitude

    :ivar get_amplitude: 获取给定数值所对应的幅值
    :vartype get_amplitude: Callable

    :ivar bc_node_ids: 边界节点编号列表
    :vartype bc_node_ids: np.ndarray

    :ivar bc_element_ids: 边界单元编号列表
    :vartype bc_element_ids: np.ndarray

    :ivar bc_dof_ids: 自由度编号列表
    :vartype bc_dof_ids: np.ndarray

    :ivar bc_surface: 边界表面
    :vartype bc_surface: list[tuple[int, str]]

    :ivar bc_dof_values: 自由度数值列表
    :vartype bc_dof_values: np.ndarray

    :ivar bc_fext: 等效节点力列表
    :vartype bc_fext: np.ndarray
    """

    __slots_dict__: dict = {
        'bc': ('BC', '边界条件属性'),
        'dof': ('Dof', '自由度属性'),
        'mesh_data': ('MeshData', '网格对象'),
        'solver': ('Solver', '求解器属性'),
        'amplitude': ('Optional[Amplitude]', '幅值属性'),
        'amplitude_data': ('BaseAmplitude', '幅值对象'),
        'get_amplitude': ('Callable', '获取给定数值所对应的幅值'),
        'bc_node_ids': ('np.ndarray', '边界节点编号列表'),
        'bc_element_ids': ('np.ndarray', '边界单元编号列表'),
        'bc_dof_ids': ('np.ndarray', '自由度编号列表'),
        'bc_surface': ('list[tuple[int, str]]', '边界表面'),
        'bc_dof_values': ('np.ndarray', '自由度数值列表'),
        'bc_fext': ('np.ndarray', '等效节点力列表')
    }

    __slots__: list = [slot for slot in __slots_dict__.keys()]

    def __init__(self, bc: BC, dof: Dof, mesh_data: MeshData, solver: Solver, amplitude: Optional[Amplitude]) -> None:
        self.bc: BC = bc
        self.dof: Dof = dof
        self.mesh_data: MeshData = mesh_data
        self.solver: Solver = solver
        self.amplitude: Optional[Amplitude] = amplitude
        if self.amplitude is not None:
            self.amplitude_data: AmplitudeData = get_amplitude_data(self.amplitude)
        else:
            self.amplitude_data = BaseAmplitude()
            self.amplitude_data.set_f_amplitude([0, solver.total_time], [0, 1])
        self.get_amplitude: Callable = self.amplitude_data.get_amplitude
        self.bc_node_ids: np.ndarray = np.empty(0)
        self.bc_element_ids: np.ndarray = np.empty(0)
        self.bc_surface: list[tuple[int, str]] = list()
        self.bc_dof_ids: np.ndarray = np.empty(0, dtype='int32')
        self.bc_dof_values: np.ndarray = np.empty(0)
        self.bc_fext: np.ndarray = np.empty(0)

    def to_string(self, level: int = 1) -> str:
        return object_slots_to_string_ndarray(self, level)

    def show(self) -> None:
        print(self.to_string())

    def get_surface_from_bc_element(self, bc_element_id: int, bc_element: np.ndarray) -> list[tuple[int, str]]:
        nodes = self.mesh_data.nodes
        elements = self.mesh_data.elements
        element_surface = []
        for element_id, element in enumerate(elements):
            is_element_surface = all(np.isin(bc_element, element))
            if is_element_surface:
                nodes_in_element = np.isin(element, bc_element)
                connectivity = elements[element_id]
                node_coords = nodes[connectivity]
                iso_element_type = get_iso_element_type(node_coords)
                iso_element_shape = iso_element_shape_dict[iso_element_type]
                surface_names = [surface_name for surface_name, nodes_on_surface in iso_element_shape.nodes_on_surface_dict.items() if
                                 all(nodes_on_surface == nodes_in_element)]
                if len(surface_names) == 1:
                    element_surface.append((element_id, surface_names[0]))
                else:
                    raise ValueError(error_style(f'the surface of element {element_id} is wrong'))

        if len(element_surface) == 1:
            return element_surface
        else:
            raise ValueError(error_style(f'the surface of bc_element {bc_element_id} is wrong'))

    def get_surface_from_elements_nodes(self, element_id: int, node_ids: list[int]) -> list[tuple[int, str]]:
        nodes = self.mesh_data.nodes
        elements = self.mesh_data.elements
        element_surface = []
        nodes_in_element = np.isin(elements[element_id], node_ids)
        connectivity = elements[element_id]
        node_coords = nodes[connectivity]
        iso_element_type = get_iso_element_type(node_coords)
        iso_element_shape = iso_element_shape_dict[iso_element_type]

        surface_names = [surface_name for surface_name, nodes_on_surface in iso_element_shape.nodes_on_surface_dict.items() if
                         sum(np.logical_and(nodes_in_element, nodes_on_surface)) == len(iso_element_shape.bc_surface_nodes_dict[surface_name])]

        for surface_name in surface_names:
            element_surface.append((element_id, surface_name))

        if 1 <= len(element_surface) <= iso_element_shape.bc_surface_number:
            return element_surface
        else:
            raise ValueError(error_style(f'the surface of element {element_id} is wrong'))


if __name__ == "__main__":
    from pyfem.utils.visualization import print_slots_dict

    print_slots_dict(BaseBC.__slots_dict__)
