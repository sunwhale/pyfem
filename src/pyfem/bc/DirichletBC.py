# -*- coding: utf-8 -*-
"""

"""
from typing import Optional

from numpy import array

from pyfem.bc.BaseBC import BaseBC
from pyfem.io.Amplitude import Amplitude
from pyfem.io.BC import BC
from pyfem.io.Dof import Dof
from pyfem.io.Solver import Solver
from pyfem.mesh.MeshData import MeshData
from pyfem.utils.colors import error_style


class DirichletBC(BaseBC):
    __slots__ = BaseBC.__slots__ + ()

    def __init__(self, bc: BC, dof: Dof, mesh_data: MeshData, solver: Solver, amplitude: Optional[Amplitude]) -> None:
        super().__init__(bc, dof, mesh_data, solver, amplitude)
        self.create_dof_values()

    def create_dof_values(self) -> None:
        bc_node_sets = self.bc.node_sets
        bc_node_ids = []
        for bc_node_set in bc_node_sets:
            bc_node_ids += list(self.mesh_data.node_sets[bc_node_set])

        # 如果发现施加当前边界条件的点集中有重复的点则抛出异常
        if len(bc_node_ids) != len(set(bc_node_ids)):
            error_msg = f'{type(self).__name__} {self.bc.name} contains repeat nodes\n'
            error_msg += f'Please check the input file'
            raise NotImplementedError(error_style(error_msg))
        else:
            self.bc_node_ids = array(bc_node_ids)

        # 确定施加的边界条件对应的全局自由度编号
        bc_dof_names = self.bc.dof
        dof_names = self.dof.names
        dof_ids = []
        for node_index in self.bc_node_ids:
            for _, bc_dof_name in enumerate(bc_dof_names):
                dof_ids.append(node_index * len(dof_names) + dof_names.index(bc_dof_name))
        self.dof_ids = array(dof_ids)

        bc_value = self.bc.value
        if isinstance(bc_value, float):
            self.dof_values = array([bc_value for _ in self.dof_ids])


if __name__ == "__main__":
    from pyfem.io.Properties import Properties

    props = Properties()
    props.read_file(r'..\..\..\examples\mechanical\plane\Job-1.toml')
    props.verify()

    bc_data = DirichletBC(props.bcs[1], props.dof, props.mesh_data, props.solver, props.amplitudes[0])
    bc_data.create_dof_values()
    bc_data.show()
