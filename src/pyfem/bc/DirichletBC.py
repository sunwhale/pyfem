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
    r"""
    **Dirichlet边界条件**

    基于边界条件的属性、自由度属性、网格对象、求解器属性和幅值属性计算系统线性方程组 :math:`{\mathbf{K u}} = {\mathbf{f}}` 中对应自由度 :math:`{\mathbf{u}}` 的约束信息。

    Dirichlet边界条件只能施加于边界条件属性中的节点集合 :py:attr:`pyfem.io.BC.BC.node_sets` 。

    对象创建时更新自由度序号列表 :py:attr:`bc_node_ids` 和对应自由度取值列表 :py:attr:`bc_dof_values` 。
    """

    __slots__ = BaseBC.__slots__ + []

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
            error_msg = f'{type(self).__name__} \'{self.bc.name}\' contains repeat nodes\n'
            error_msg += f'Please check the input file'
            raise NotImplementedError(error_style(error_msg))
        else:
            self.bc_node_ids = array(bc_node_ids)

        # 确定施加的边界条件对应的全局自由度编号
        bc_dof_names = self.bc.dof
        dof_names = self.dof.names
        bc_dof_ids = []
        for node_index in self.bc_node_ids:
            for _, bc_dof_name in enumerate(bc_dof_names):
                bc_dof_ids.append(node_index * len(dof_names) + dof_names.index(bc_dof_name))
        self.bc_dof_ids = array(bc_dof_ids, dtype='int32')

        bc_value = self.bc.value
        if isinstance(bc_value, float) or isinstance(bc_value, int):
            self.bc_dof_values = array([bc_value for _ in self.bc_dof_ids])
        else:
            error_msg = f'in {type(self).__name__} \'{self.bc.name}\' the value of \'{bc_value}\' is not a float or int number'
            raise NotImplementedError(error_style(error_msg))


if __name__ == "__main__":
    from pyfem.io.Properties import Properties

    props = Properties()
    props.read_file(r'..\..\..\examples\mechanical\plane\Job-1.toml')
    props.verify()

    bc_data = DirichletBC(props.bcs[1], props.dof, props.mesh_data, props.solver, props.amplitudes[0])
    bc_data.create_dof_values()
    bc_data.show()
