# -*- coding: utf-8 -*-
"""

"""
import re
from pathlib import Path
from typing import Dict, List, Union

try:
    import tomllib  # type: ignore
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore

from pyfem.io.Dof import Dof
from pyfem.io.Section import Section
from pyfem.io.Material import Material
from pyfem.io.Mesh import Mesh
from pyfem.io.BC import BC
from pyfem.io.Amplitude import Amplitude
from pyfem.io.Solver import Solver
from pyfem.io.Output import Output
from pyfem.mesh.MeshData import MeshData
from pyfem.utils.colors import CYAN, MAGENTA, BLUE, END, error_style
from pyfem.utils.logger import logger
from pyfem.io.BaseIO import BaseIO


class Properties(BaseIO):
    """
    有限元算例的属性类，解析配置文件中定义的算例属性。

    :ivar work_path: 工作目录
    :vartype work_path: Path

    :ivar input_file: 算例输入文件路径
    :vartype input_file: Path

    :ivar abs_input_file: 算例输入文件绝对路径
    :vartype abs_input_file: Path

    :ivar toml: toml文件解析后的字典
    :vartype toml: Dict

    :ivar title: 算例标题
    :vartype title: str

    :ivar mesh: 网格属性
    :vartype mesh: Mesh

    :ivar dof: 自由度属性
    :vartype dof: Dof

    :ivar materials: 材料属性列表
    :vartype materials: List[Material]

    :ivar sections: 截面属性列表
    :vartype sections: List[Section]

    :ivar amplitudes: 幅值属性列表
    :vartype amplitudes: List[Amplitude]

    :ivar bcs: 边界条件属性列表
    :vartype bcs: List[BC]

    :ivar solver: 求解器属性
    :vartype solver: Solver

    :ivar outputs: 输出配置属性列表
    :vartype outputs: List[Output]

    :ivar mesh_data: 网格文件解析后的网格数据
    :vartype mesh_data: MeshData

    :ivar parameter_filename: 算例标题
    :vartype parameter_filename: str

    :ivar parameters: parameters.toml文件解析后的字典
    :vartype parameters: Dict
    """

    __slots_dict__: Dict = {
        'work_path': ('Path', '工作目录'),
        'input_file': ('Path', '算例输入文件路径'),
        'abs_input_file': ('Path', '算例输入文件绝对路径'),
        'toml': ('Dict', 'toml文件解析后的字典'),
        'title': ('str', '算例标题'),
        'mesh': ('Mesh', '网格属性'),
        'dof': ('Dof', '自由度属性'),
        'materials': ('List[Material]', '材料属性列表'),
        'sections': ('List[Section]', '截面属性列表'),
        'amplitudes': ('List[Amplitude]', '幅值属性列表'),
        'bcs': ('List[BC]', '边界条件属性列表'),
        'solver': ('Solver', '求解器属性'),
        'outputs': ('List[Output]', '输出配置属性列表'),
        'mesh_data': ('MeshData', '网格文件解析后的网格数据'),
        'parameter_filename': ('str', '算例标题'),
        'parameters': ('Dict', 'parameters.toml文件解析后的字典'),
    }

    __slots__: List = [slot for slot in __slots_dict__.keys()]

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
        self.parameter_filename: str = None  # type: ignore
        self.parameters: Dict = None  # type: ignore

    def verify(self) -> None:
        logger.info('INPUT FILE VALIDATING')
        is_error = False
        error_msg = '\nInput error:\n'
        for key in [slot for slot in self.__slots__ if slot not in ['work_path', 'input_file', 'abs_input_file', 'parameter_filename']]:  # 忽略非必须的关键字
            if self.__getattribute__(key) is None:
                is_error = True
                error_msg += f'  - {key} is missing\n'

        if is_error:
            error_msg += f'Please check the input file {self.input_file}'
            raise NotImplementedError(error_msg)
        logger.info('INPUT FILE VALIDATED')

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

    def set_parameters(self, parameter_file: Path) -> None:
        with open(parameter_file, "rb") as f:
            parameters = tomllib.load(f)
            self.parameters = parameters

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
                raise NotImplementedError(error_msg)

        if 'parameter_filename' in toml_keys:
            self.parameter_filename = self.toml['parameter_filename']
            parameter_file = Path(self.parameter_filename)
            if parameter_file.is_absolute():
                abs_parameter_file = parameter_file
            else:
                abs_parameter_file = self.work_path.joinpath(parameter_file).resolve()
            self.set_parameters(abs_parameter_file)
        else:
            self.parameters = {}

        try:
            substitute_parameters(self.toml, self.parameters)
        except KeyError as e:
            error_message = f'{e} is not given in the dict of parameters, please check the parameters file'
            raise NotImplementedError(error_message)

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


def extract_parameter_label(string: str) -> str:
    """
    从带有<>的字符串中提取参数标签
    """
    pattern = r'<(.*?)>'
    matches = re.findall(pattern, string)
    return matches[0]


def substitute_parameters(data: dict | list, parameters: dict) -> None:
    """
    用参数字典中的数据替换toml字典中用<>定义的变量
    """
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                substitute_parameters(value, parameters)
            elif isinstance(value, str) and value.strip().startswith("<") and value.strip().endswith(">"):
                parameter_label = extract_parameter_label(value)
                data[key] = parameters[parameter_label]
    elif isinstance(data, list):
        for i in range(len(data)):
            if isinstance(data[i], (dict, list)):
                substitute_parameters(data[i], parameters)
            elif isinstance(data[i], str) and data[i].strip().startswith("<") and data[i].strip().endswith(">"):
                parameter_label = extract_parameter_label(data[i])
                data[i] = parameters[parameter_label]


if __name__ == "__main__":
    # from pyfem.utils.visualization import print_slots_dict
    #
    # print_slots_dict(Properties.__slots_dict__)

    props = Properties()
    props.read_file(r'..\..\..\examples\mechanical\plane\Job-1.toml')
    props.show()
