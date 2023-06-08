# -*- coding: utf-8 -*-
"""

"""
from typing import Optional

from pyfem.bc.BaseBC import BaseBC
from pyfem.bc.DirichletBC import DirichletBC
from pyfem.bc.NeumannBC import NeumannBC
from pyfem.io.BC import BC
from pyfem.io.Dof import Dof
from pyfem.io.Amplitude import Amplitude
from pyfem.mesh.MeshData import MeshData
from pyfem.utils.colors import error_style

bc_data_dict = {
    'DirichletBC': DirichletBC,
    'NeumannBC': NeumannBC
}


def get_bc_data(bc: BC,
                dof: Dof,
                mesh_data: MeshData,
                amplitude: Optional[Amplitude]) -> BaseBC:
    class_name = f'{bc.category}{bc.type}'.strip().replace(' ', '')

    if class_name in bc_data_dict:
        return bc_data_dict[class_name](bc=bc,
                                        dof=dof,
                                        mesh_data=mesh_data,
                                        amplitude=amplitude)
    else:
        error_msg = f'{class_name} bc is not supported.\n'
        error_msg += f'The allowed bc types are {list(bc_data_dict.keys())}.'
        raise NotImplementedError(error_style(error_msg))


if __name__ == "__main__":
    pass
