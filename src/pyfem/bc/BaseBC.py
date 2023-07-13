# -*- coding: utf-8 -*-
"""

"""
from typing import List, Optional, Callable, Tuple

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
    __slots__: Tuple = ('bc',
                        'dof',
                        'mesh_data',
                        'solver',
                        'amplitude',
                        'amplitude_data',
                        'get_amplitude',
                        'bc_node_ids',
                        'bc_element_ids',
                        'dof_ids',
                        'dof_values',
                        'bc_fext',
                        'bc_surface')

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
        self.bc_surface: List[Tuple[int, str]] = []

    def to_string(self, level: int = 1) -> str:
        return object_slots_to_string_ndarray(self, level)

    def show(self) -> None:
        print(self.to_string())


if __name__ == "__main__":
    pass
