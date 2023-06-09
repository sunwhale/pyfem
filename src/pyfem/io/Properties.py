# -*- coding: utf-8 -*-
"""

"""
from pathlib import Path
from typing import Dict, List, Any, Tuple, Union

try:
    import tomllib  # type: ignore
except ModuleNotFoundError:
    import tomli as tomllib

from pyfem.io.Dof import Dof
from pyfem.io.Section import Section
from pyfem.io.Material import Material
from pyfem.io.Mesh import Mesh
from pyfem.io.BC import BC
from pyfem.io.Amplitude import Amplitude
from pyfem.io.Solver import Solver
from pyfem.io.Output import Output
from pyfem.mesh.MeshData import MeshData
from pyfem.utils.colors import CYAN, MAGENTA, BLUE, END, error_style, info_style
from pyfem.io.BaseIO import BaseIO


class Properties(BaseIO):
    """
    Properties类用于解析配置文件中定义的属性。
    当 self.is_read_only = True 时：

    1. Properties 类的所有属性在首次被赋非None值后不能再被修改和删除，

    2. 此时许可的属性关键字存储在self.slots中。
    """
    __slots__: Tuple = ('work_path',
                        'input_file',
                        'abs_input_file',
                        'toml',
                        'title',
                        'mesh',
                        'dof',
                        'materials',
                        'sections',
                        'amplitudes',
                        'bcs',
                        'solver',
                        'outputs',
                        'mesh_data')

    def __init__(self) -> None:
        super().__init__()
        self.work_path: Path = None  # type: ignore
        self.input_file: Path = None  # type: ignore
        self.abs_input_file: Path = None  # type: ignore
        self.toml: Dict = None  # type: ignore
        self.title: str = None  # type: ignore
        self.mesh: Mesh = None  # type: ignore
        self.dof: Dof = None  # type: ignore
        self.materials: List[Material] = None  # type: ignore
        self.sections: List[Section] = None  # type: ignore
        self.amplitudes: List[Amplitude] = None  # type: ignore
        self.bcs: List[BC] = None  # type: ignore
        self.solver: Solver = None  # type: ignore
        self.outputs: List[Output] = None  # type: ignore
        self.mesh_data: MeshData = None  # type: ignore

    def verify(self) -> None:
        print(info_style('Verifying the input ...'))
        is_error = False
        error_msg = '\nInput error:\n'
        for key in self.__slots__[3:]:  # 忽略这3个关键字：'work_path', 'input_file', 'abs_input_file'，它们不是在.toml中定义的
            if self.__getattribute__(key) is None:
                is_error = True
                error_msg += f'  - {key} is missing\n'

        if is_error:
            error_msg += f'Please check the input file {self.input_file}'
            raise NotImplementedError(error_style(error_msg))

    def show(self) -> None:
        for key in self.__slots__:
            item = self.__getattribute__(key)
            print()
            print(CYAN + f'+-{key}' + END)
            print(MAGENTA + f'  |- {type(item)}' + END)
            if hasattr(item, 'to_string'):
                print(f'  |- {item.to_string()}')
            else:
                print(f'  |- {item}')
            if isinstance(item, list):
                for i, it in enumerate(item):
                    print(BLUE + f'    |-{i}-{it.to_string(level=3)}' + END)

    def set_toml(self, toml: Dict) -> None:
        self.toml = toml

    def set_title(self, title: str) -> None:
        self.title = title

    def set_mesh(self, mesh_dict: Dict) -> None:
        self.mesh = Mesh()
        self.mesh.set_io_values(mesh_dict)

        mesh_path = Path(self.mesh.file)
        if mesh_path.is_absolute():  # 判断 self.mesh.file 是不是绝对路径
            abs_mesh_path = mesh_path
        else:  # 如果 self.mesh.file 不是绝对路径，则用工作目录 self.work_path 补全为绝对路径
            abs_mesh_path = self.work_path.joinpath(mesh_path)
        self.mesh_data = MeshData()
        self.mesh_data.read_file(abs_mesh_path, self.mesh.type)

    def set_dofs(self, dofs_dict: Dict) -> None:
        self.dof = Dof()
        self.dof.set_io_values(dofs_dict)

    def set_solver(self, solver_dict: Dict) -> None:
        self.solver = Solver()
        self.solver.set_io_values(solver_dict)

    def set_materials(self, materials_list: List) -> None:
        self.materials = []
        for material_dict in materials_list:
            material = Material()
            material.set_io_values(material_dict)
            self.materials.append(material)

    def set_sections(self, sections_list: List) -> None:
        self.sections = []
        for section_dict in sections_list:
            section = Section()
            section.set_io_values(section_dict)
            self.sections.append(section)

    def set_amplitudes(self, amplitudes_list: List) -> None:
        self.amplitudes = []
        for amplitude_dict in amplitudes_list:
            amplitude = Amplitude()
            amplitude.set_io_values(amplitude_dict)
            self.amplitudes.append(amplitude)

    def set_bcs(self, bcs_list: List) -> None:
        self.bcs = []
        for bc_dict in bcs_list:
            bc = BC()
            bc.set_io_values(bc_dict)
            self.bcs.append(bc)

    def set_outputs(self, outputs_list: List) -> None:
        self.outputs = []
        for output_dict in outputs_list:
            output = Output()
            output.set_io_values(output_dict)
            self.outputs.append(output)

    def read_file(self, filename: Union[Path, str]) -> None:
        """
        读取 .toml 格式的配置文件。
        """
        self.input_file = Path(filename)
        self.work_path = self.input_file.parent
        if self.input_file.is_absolute():
            self.abs_input_file = self.input_file
        else:
            self.abs_input_file = Path.cwd().joinpath(self.input_file).resolve()

        with open(self.input_file, "rb") as f:
            toml = tomllib.load(f)
            self.set_toml(toml)

        toml_keys = self.toml.keys()
        allowed_keys = self.__slots__

        for key in toml_keys:
            if key not in allowed_keys:
                error_msg = f'{key} is not an allowable attribute keyword of {type(self).__name__}\n'
                error_msg += f'please check the file {self.abs_input_file}'
                raise AttributeError(error_style(error_msg))

        if 'title' in toml_keys:
            title = self.toml['title']
            self.set_title(title)

        if 'sections' in toml_keys:
            sections_list = self.toml['sections']
            self.set_sections(sections_list)

        if 'mesh' in toml_keys:
            mesh_dict = self.toml['mesh']
            self.set_mesh(mesh_dict)

        if 'dof' in toml_keys:
            dofs_dict = self.toml['dof']
            self.set_dofs(dofs_dict)

        if 'materials' in toml_keys:
            materials_list = self.toml['materials']
            self.set_materials(materials_list)

        if 'amplitudes' in toml_keys:
            amplitudes_list = self.toml['amplitudes']
            self.set_amplitudes(amplitudes_list)

        if 'bcs' in toml_keys:
            bcs_list = self.toml['bcs']
            self.set_bcs(bcs_list)

        if 'solver' in toml_keys:
            solver_dict = self.toml['solver']
            self.set_solver(solver_dict)

        if 'outputs' in toml_keys:
            outputs_dict = self.toml['outputs']
            self.set_outputs(outputs_dict)

        self.verify()


if __name__ == "__main__":
    props = Properties()
    props.read_file(r'..\..\..\examples\mechanical\plane\Job-1.toml')
    props.show()

    # props.outputs[0].type = 1
