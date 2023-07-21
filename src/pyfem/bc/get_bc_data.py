# -*- coding: utf-8 -*-
"""

"""
from typing import Union, Optional

from pyfem.bc.BaseBC import BaseBC
from pyfem.bc.DirichletBC import DirichletBC
from pyfem.bc.NeumannBCConcentrated import NeumannBCConcentrated
from pyfem.bc.NeumannBCDistributed import NeumannBCDistributed
from pyfem.io.Amplitude import Amplitude
from pyfem.io.BC import BC
from pyfem.io.Dof import Dof
from pyfem.io.Solver import Solver
from pyfem.mesh.MeshData import MeshData
from pyfem.utils.colors import error_style

BCData = Union[BaseBC, DirichletBC, NeumannBCConcentrated, NeumannBCDistributed]

bc_data_dict = {
    'DirichletBC': DirichletBC,
    'NeumannBCConcentrated': NeumannBCConcentrated,
    'NeumannBCDistributed': NeumannBCDistributed
}


def get_bc_data(bc: BC,
                dof: Dof,
                mesh_data: MeshData,
                solver: Solver,
                amplitude: Optional[Amplitude]) -> BCData:
    """
    工厂函数，用于根据边界条件属性生产不同的边界条件对象。

    Args:
        bc(BC): 边界条件属性
        dof(Dof): 自由度属性
        mesh_data(MeshData): 网格数据对象
        solver(Solver): 求解器属性
        amplitude(Optional[Amplitude]): 幅值属性

    :return: 边界条件对象
    :rtype: BCData
    """

    class_name = f'{bc.category}{bc.type}'.strip().replace(' ', '')

    if class_name in bc_data_dict:
        return bc_data_dict[class_name](bc=bc,
                                        dof=dof,
                                        mesh_data=mesh_data,
                                        solver=solver,
                                        amplitude=amplitude)
    else:
        error_msg = f'{class_name} bc is not supported.\n'
        error_msg += f'The allowed bc types are {list(bc_data_dict.keys())}.'
        raise NotImplementedError(error_style(error_msg))


if __name__ == "__main__":
    pass
