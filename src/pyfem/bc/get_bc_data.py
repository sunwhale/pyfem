from pyfem.bc.BaseBC import BaseBC
from pyfem.bc.DirichletBC import DirichletBC
from pyfem.io.BC import BC
from pyfem.io.Dof import Dof
from pyfem.mesh.NodeSet import NodeSet
from pyfem.utils.colors import error_style

bc_data_dict = {
    'DirichletBC': DirichletBC
}


def get_bc_data(bc: BC,
                dof: Dof,
                nodes: NodeSet) -> BaseBC:
    class_name = f'{bc.type}'.strip().replace(' ', '')

    if class_name in bc_data_dict:
        return bc_data_dict[class_name](bc=bc,
                                        dof=dof,
                                        nodes=nodes)
    else:
        error_msg = f'{class_name} bc is not supported.\n'
        error_msg += f'The allowed bc types are {list(bc_data_dict.keys())}.'
        raise NotImplementedError(error_style(error_msg))


if __name__ == "__main__":
    pass
