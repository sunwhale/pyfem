# -*- coding: utf-8 -*-
"""

"""
import time

from numpy import repeat, array, ndarray, empty, zeros
from scipy.sparse import coo_matrix, csc_matrix  # type: ignore

from pyfem.bc.get_bc_data import get_bc_data, BCData
from pyfem.elements.get_element_data import get_element_data, ElementData
from pyfem.fem.Timer import Timer
from pyfem.fem.constants import DTYPE, IS_PETSC
from pyfem.io.Amplitude import Amplitude
from pyfem.io.Material import Material
from pyfem.io.Properties import Properties
from pyfem.io.Section import Section
from pyfem.isoelements.IsoElementShape import iso_element_shape_dict
from pyfem.isoelements.get_iso_element_type import get_iso_element_type
from pyfem.materials.get_material_data import get_material_data
from pyfem.utils.colors import error_style
from pyfem.utils.visualization import object_slots_to_string_assembly
from pyfem.utils.wrappers import show_running_time


class Assembly:
    """
    定义装配体。

    :ivar total_dof_number: 总自由度数量
    :vartype total_dof_number: int

    :ivar props: 作业属性对象
    :vartype props: Properties

    :ivar timer: 计时器对象
    :vartype timer: Timer

    :ivar materials_dict: 材料字典：材料名称->材料属性对象
    :vartype materials_dict: dict[str, Material]

    :ivar sections_dict: 截面字典：截面名称->截面属性对象
    :vartype sections_dict: dict[str, Section]

    :ivar amplitudes_dict: 幅值字典：幅值名称->幅值属性对象
    :vartype amplitudes_dict: dict[str, Amplitude]

    :ivar section_of_element_set: 单元集合截面字典：单元集合名称->截面属性对象
    :vartype section_of_element_set: dict[str, Section]

    :ivar element_data_list: 单元数据对象列表
    :vartype element_data_list: list[ElementData]

    :ivar bc_data_list: 边界条件数据对象列表
    :vartype bc_data_list: list[BCData]

    :ivar global_stiffness: 全局刚度矩阵
    :vartype global_stiffness: csc_matrix(total_dof_number, total_dof_number)

    :ivar fext: 等式右边外力向量
    :vartype fext: ndarray(total_dof_number,)

    :ivar fint: 内力向量
    :vartype fint: ndarray(total_dof_number,)

    :ivar dof_solution: 全局自由度的值
    :vartype dof_solution: ndarray(total_dof_number,)

    :ivar ddof_solution: 全局自由度增量的值
    :vartype ddof_solution: ndarray(total_dof_number,)

    :ivar bc_dof_ids: 边界自由度列表
    :vartype bc_dof_ids: ndarray

    :ivar field_variables: 常变量字典
    :vartype field_variables: dict[str, ndarray]
    """

    __slots_dict__: dict = {
        'total_dof_number': ('int', '总自由度数量'),
        'props': ('Properties', '作业属性对象'),
        'timer': ('Timer', '计时器对象'),
        'materials_dict': ('dict[str, Material]', '材料字典：材料名称->材料属性对象'),
        'sections_dict': ('dict[str, Section]', '截面字典：截面名称->截面属性对象'),
        'amplitudes_dict': ('dict[str, Amplitude]', '幅值字典：幅值名称->幅值属性对象'),
        'section_of_element_set': ('dict[str, Section]', '单元集合截面字典：单元集合名称->截面属性对象'),
        'element_data_list': ('list[ElementData]', '单元数据对象列表'),
        'bc_data_list': ('list[BCData]', '边界条件数据对象列表'),
        'global_stiffness': ('csc_matrix(total_dof_number, total_dof_number)', '全局刚度矩阵'),
        'fext': ('ndarray(total_dof_number,)', '等式右边外力向量'),
        'fint': ('ndarray(total_dof_number,)', '内力向量'),
        'dof_solution': ('ndarray(total_dof_number,)', '全局自由度的值'),
        'ddof_solution': ('ndarray(total_dof_number,)', '全局自由度增量的值'),
        'bc_dof_ids': ('ndarray', '边界自由度列表'),
        'field_variables': ('dict[str, ndarray]', '常变量字典'),
        'A': ('A', 'A')
    }

    __slots__: list = [slot for slot in __slots_dict__.keys()]

    def __init__(self, props: Properties) -> None:
        self.total_dof_number: int = -1
        self.props: Properties = props
        self.timer: Timer = Timer()
        self.materials_dict: dict[str, Material] = dict()
        self.sections_dict: dict[str, Section] = dict()
        self.amplitudes_dict: dict[str, Amplitude] = dict()
        self.section_of_element_set: dict[str, Section] = dict()
        self.element_data_list: list[ElementData] = list()
        self.bc_data_list: list[BCData] = list()
        self.global_stiffness: csc_matrix = csc_matrix(0)
        self.fext: ndarray = empty(0, dtype=DTYPE)
        self.fint: ndarray = empty(0, dtype=DTYPE)
        self.dof_solution: ndarray = empty(0, dtype=DTYPE)
        self.ddof_solution: ndarray = empty(0, dtype=DTYPE)
        self.bc_dof_ids: ndarray = empty(0)
        self.field_variables: dict[str, ndarray] = dict()
        self.init()
        self.update_element_data()
        self.assembly_global_stiffness()

    def to_string(self, level: int = 1) -> str:
        return object_slots_to_string_assembly(self, level)

    def show(self) -> None:
        print(self.to_string())

    @show_running_time
    def init(self) -> None:
        # 初始化 self.element_data_list
        mesh_data = self.props.mesh_data
        elements = mesh_data.elements
        nodes = mesh_data.nodes
        element_sets = mesh_data.element_sets
        sections = self.props.sections
        materials = self.props.materials
        solver = self.props.solver
        amplitudes = self.props.amplitudes
        dof = self.props.dof
        timer = self.timer
        dimension = self.props.mesh_data.dimension
        self.total_dof_number = len(nodes) * len(dof.names)

        for material in materials:
            self.materials_dict[material.name] = material

        for section in sections:
            self.sections_dict[section.name] = section

        for amplitude in amplitudes:
            self.amplitudes_dict[amplitude.name] = amplitude

        for element_set in element_sets:
            for section in sections:
                if element_set in section.element_sets:
                    self.section_of_element_set[element_set] = section

        for element_set_name, element_ids in element_sets.items():
            if element_set_name in self.section_of_element_set.keys():
                section = self.section_of_element_set[element_set_name]
                materials = [self.materials_dict[material_name] for material_name in section.material_names]
                material_data_list = [get_material_data(material=material,
                                                        dimension=dimension,
                                                        section=section) for material in materials]

                for element_id in element_ids:
                    connectivity = elements[element_id]
                    node_coords = nodes[connectivity]
                    iso_element_type = get_iso_element_type(node_coords)
                    iso_element_shape = iso_element_shape_dict[iso_element_type]
                    element_data = get_element_data(element_id=element_id,
                                                    iso_element_shape=iso_element_shape,
                                                    connectivity=connectivity,
                                                    node_coords=node_coords,
                                                    dof=dof,
                                                    materials=materials,
                                                    section=section,
                                                    material_data_list=material_data_list,
                                                    timer=timer)

                    element_data.assembly_conn = connectivity
                    element_data.create_element_dof_ids()

                    self.element_data_list.append(element_data)

        if len(self.element_data_list) < len(self.props.mesh_data.elements):
            raise NotImplementedError(error_style(f'some elements do not have defined section properties'))
        elif len(self.element_data_list) > len(self.props.mesh_data.elements):
            raise NotImplementedError(
                error_style(f'some elements have section properties that are redundantly defined'))

        # 初始化 self.bc_data_list
        bcs = self.props.bcs
        for bc in bcs:
            if bc.amplitude_name is not None:
                amplitude = self.amplitudes_dict[bc.amplitude_name]
            else:
                amplitude = None
            bc_data = get_bc_data(bc=bc,
                                  dof=dof,
                                  mesh_data=mesh_data,
                                  solver=solver,
                                  amplitude=amplitude)
            self.bc_data_list.append(bc_data)

        bc_dof_ids = []
        for bc_data in self.bc_data_list:
            for bc_dof_id, bc_dof_value in zip(bc_data.bc_dof_ids, bc_data.bc_dof_values):
                bc_dof_ids.append(bc_dof_id)
        self.bc_dof_ids = array(bc_dof_ids)

        # 初始化 rhs, fext, fint, dof_solution, ddof_solution
        self.fext = zeros(self.total_dof_number, dtype=DTYPE)
        self.fint = zeros(self.total_dof_number, dtype=DTYPE)
        self.dof_solution = zeros(self.total_dof_number, dtype=DTYPE)
        self.ddof_solution = zeros(self.total_dof_number, dtype=DTYPE)

    # @show_running_time
    def assembly_global_stiffness(self) -> None:
        if IS_PETSC:
            try:
                from petsc4py import PETSc
            except:
                raise ImportError(error_style('petsc4py can not be imported'))

            self.A = PETSc.Mat()
            self.A.createAIJ((self.total_dof_number, self.total_dof_number))
            for element_data in self.element_data_list:
                element_dof_ids = element_data.element_dof_ids
                self.A.setValues(element_dof_ids, element_dof_ids, element_data.element_stiffness, addv=True)
            self.A.assemble()

        else:
            val = []
            row = []
            col = []

            for element_data in self.element_data_list:
                element_dof_ids = element_data.element_dof_ids
                element_dof_number = element_data.element_dof_number
                row += [r for r in repeat(element_dof_ids, element_dof_number)]
                for _ in range(element_dof_number):
                    col += [c for c in element_dof_ids]
                val += [v for v in element_data.element_stiffness.reshape(element_dof_number * element_dof_number)]

            self.global_stiffness = coo_matrix((array(val, dtype=DTYPE), (array(row, dtype=DTYPE), array(col, dtype=DTYPE))), shape=(self.total_dof_number, self.total_dof_number)).tocsr()

    # @show_running_time
    def assembly_fint(self) -> None:
        self.fint = zeros(self.total_dof_number, dtype=DTYPE)
        for element_data in self.element_data_list:
            self.fint[element_data.element_dof_ids] += element_data.element_fint

    # @show_running_time
    def update_element_data(self) -> None:
        dof_solution = self.dof_solution
        ddof_solution = self.ddof_solution
        for element_data in self.element_data_list:
            element_data.update_element_dof_values(dof_solution)
            element_data.update_element_ddof_values(ddof_solution)
            element_data.update_element_material_stiffness_fint()

    # @show_running_time
    def update_element_data_without_stiffness(self) -> None:
        dof_solution = self.dof_solution
        ddof_solution = self.ddof_solution
        for element_data in self.element_data_list:
            element_data.update_element_dof_values(dof_solution)
            element_data.update_element_ddof_values(ddof_solution)
            element_data.update_element_material_stiffness_fint(is_update_stiffness=False)

    # @show_running_time
    def update_element_state_variables(self) -> None:
        for element_data in self.element_data_list:
            element_data.update_element_state_variables()

    def goback_element_state_variables(self) -> None:
        for element_data in self.element_data_list:
            element_data.goback_element_state_variables()

    # @show_running_time
    def update_element_field_variables(self) -> None:
        for element_data in self.element_data_list:
            element_data.update_element_field_variables()

    def assembly_field_variables(self) -> None:
        nodes_number = len(self.props.mesh_data.nodes)

        for output in self.props.outputs:
            if output.type == 'vtk':
                for field_name in output.field_outputs:
                    self.field_variables[field_name] = zeros(nodes_number)
                    nodes_count = zeros(nodes_number)
                    for element_data in self.element_data_list:
                        assembly_conn = element_data.assembly_conn
                        self.field_variables[field_name][assembly_conn] += \
                            element_data.element_average_field_variables[field_name]
                        nodes_count[assembly_conn] += 1.0
                    self.field_variables[field_name] = self.field_variables[field_name] / nodes_count


if __name__ == "__main__":
    from pyfem.utils.visualization import print_slots_dict

    print_slots_dict(Assembly.__slots_dict__)

    from pyfem.io.Properties import Properties

    props = Properties()
    props.read_file(r'..\..\..\examples\mechanical\plane\Job-1.toml')
    assembly = Assembly(props)
    assembly.show()
