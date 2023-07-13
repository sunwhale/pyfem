# -*- coding: utf-8 -*-
"""

"""
from pathlib import Path
from typing import Tuple, Union

from pyfem.assembly.Assembly import Assembly
from pyfem.io.Properties import Properties
from pyfem.solvers.get_solver_data import get_solver_data
from pyfem.utils.visualization import object_slots_to_string


class Job:
    __slots__: Tuple = ('input_file', 'work_directory', 'abs_input_file', 'props', 'assembly', 'solver_data')

    def __init__(self, filename: Union[Path, str]) -> None:
        input_file = Path(filename)
        if input_file.is_absolute():
            abs_input_file = input_file
        else:
            abs_input_file = Path.cwd().joinpath(input_file).resolve()
        self.input_file = input_file
        self.work_directory = Path.cwd()
        self.abs_input_file = abs_input_file
        self.props = Properties()
        self.props.read_file(abs_input_file)
        # self.props.show()
        self.assembly = Assembly(self.props)
        self.solver_data = get_solver_data(self.assembly, self.props.solver)

    def run(self) -> int:
        return self.solver_data.run()

    def to_string(self, level: int = 1) -> str:
        return object_slots_to_string(self, level)

    def show(self) -> None:
        print(self.to_string())


if __name__ == '__main__':
    job = Job(r'..\..\examples\mechanical\plane\Job-1.toml')
    job.show()
