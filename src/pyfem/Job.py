# -*- coding: utf-8 -*-
"""

"""
from pathlib import Path
from typing import Union

from pyfem.assembly.Assembly import Assembly
from pyfem.io.Properties import Properties
from pyfem.solvers.get_solver_data import get_solver_data, SolverData
from pyfem.utils.visualization import object_slots_to_string


class Job:
    """
    求解器基类。

    :ivar input_file: 输入文件路径
    :vartype input_file: Path

    :ivar work_directory: 工作目录
    :vartype work_directory: Path

    :ivar abs_input_file: 输入文件绝对路径
    :vartype abs_input_file: Path

    :ivar props: 属性对象
    :vartype props: Properties

    :ivar assembly: 装配体属性
    :vartype assembly: Assembly

    :ivar solver_data: 求解器对象
    :vartype solver_data: SolverData
    """

    __slots_dict__: dict = {
        'input_file': ('Path', '输入文件路径'),
        'work_directory': ('Path', '工作目录'),
        'abs_input_file': ('Path', '输入文件绝对路径'),
        'props': ('Properties', '属性对象'),
        'assembly': ('Assembly', '装配体属性'),
        'solver_data': ('SolverData', '求解器对象')
    }

    __slots__: list = [slot for slot in __slots_dict__.keys()]

    def __init__(self, filename: Union[Path, str]) -> None:
        input_file = Path(filename)
        if input_file.is_absolute():
            abs_input_file = input_file
        else:
            abs_input_file = Path.cwd().joinpath(input_file).resolve()
        self.input_file: Path = input_file
        self.work_directory: Path = Path.cwd()
        self.abs_input_file: Path = abs_input_file
        self.props: Properties = Properties()
        self.props.read_file(abs_input_file)
        self.assembly: Assembly = Assembly(self.props)
        self.solver_data: SolverData = get_solver_data(self.assembly, self.props.solver)

    def run(self) -> int:
        return self.solver_data.run()

    def to_string(self, level: int = 1) -> str:
        return object_slots_to_string(self, level)

    def show(self) -> None:
        print(self.to_string())


if __name__ == '__main__':
    from pyfem.utils.visualization import print_slots_dict

    print_slots_dict(Job.__slots_dict__)

    job = Job(r'..\..\examples\mechanical\plane\Job-1.toml')
    job.show()
