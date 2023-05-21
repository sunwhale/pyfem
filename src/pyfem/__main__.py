# -*- coding: utf-8 -*-
"""

"""
from pathlib import Path

from pyfem.assembly.Assembly import Assembly
from pyfem.io.Properties import Properties
from pyfem.io.arguments import get_arguments
from pyfem.io.write_vtk import write_vtk
from pyfem.solvers.get_solver_data import get_solver_data
from pyfem.utils.wrappers import show_running_time


@show_running_time
def main(base_path: Path = Path.cwd()) -> None:
    args = get_arguments()

    props = Properties()

    props.base_path = base_path

    input_file = Path(args.i)

    if input_file.is_absolute():
        abs_input_file = input_file
    else:
        abs_input_file = base_path.joinpath(input_file)

    props.read_file(abs_input_file)

    props.verify()

    props.show()

    assembly = Assembly(props)

    solver_data = get_solver_data(assembly, props.solver)

    solver_data.run()

    write_vtk(props, assembly)
