# -*- coding: utf-8 -*-
"""

"""
from pathlib import Path
from typing import Union

from pyfem.assembly.Assembly import Assembly
from pyfem.io.Properties import Properties
from pyfem.io.write_vtk import write_vtk
from pyfem.solvers.get_solver_data import get_solver_data
from pyfem.utils.visualization import object_dict_to_string


class Job:
    def __init__(self, filename: Union[Path, str]) -> None:
        input_file = Path(filename)
        if input_file.is_absolute():
            abs_input_file = input_file
        else:
            abs_input_file = Path.cwd().joinpath(input_file)
        self.props = Properties()
        self.props.read_file(abs_input_file)
        # self.props.show()
        self.assembly = Assembly(self.props)
        self.solver_data = get_solver_data(self.assembly, self.props.solver)

    def run(self):
        self.solver_data.run()
        write_vtk(self.props, self.assembly)

    def to_string(self, level: int = 1) -> str:
        return object_dict_to_string(self, level)

    def show(self) -> None:
        print(self.to_string())
