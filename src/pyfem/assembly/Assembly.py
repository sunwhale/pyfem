# -*- coding: utf-8 -*-
"""

"""
from typing import List, Dict
from copy import deepcopy

from numpy import repeat, array, ndarray, empty, zeros
from scipy.sparse import coo_matrix, csc_matrix  # type: ignore

from pyfem.bc.BaseBC import BaseBC
from pyfem.bc.get_bc_data import get_bc_data
from pyfem.elements.BaseElement import BaseElement
from pyfem.elements.IsoElementShape import IsoElementShape
from pyfem.elements.get_element_data import get_element_data
from pyfem.elements.get_iso_element_type import get_iso_element_type
from pyfem.io.Properties import Properties
from pyfem.materials.get_material_data import get_material_data
from pyfem.utils.visualization import object_dict_to_string_assembly
from pyfem.utils.wrappers import show_running_time

iso_element_shape_dict = {
    'line2': IsoElementShape('line2'),
    'line3': IsoElementShape('line3'),
    'tria3': IsoElementShape('tria3'),
    'quad4': IsoElementShape('quad4'),
    'quad8': IsoElementShape('quad8'),
    'tetra4': IsoElementShape('tetra4'),
    'hex8': IsoElementShape('hex8')
}


class Assembly:
    def __init__(self, props: Properties) -> None:
        self.total_dof_number: int = -1
        self.props: Properties = props
        self.materials_dict: Dict = {}
        self.sections_dict: Dict = {}
        self.section_of_element_set: Dict = {}
        self.element_data_list: List[BaseElement] = []
        self.bc_data_list: List[BaseBC] = []
        self.global_stiffness: csc_matrix = csc_matrix(0)
        self.rhs: ndarray = empty(0)
        self.fext: ndarray = empty(0)
        self.fint: ndarray = empty(0)
        self.dof_solution: ndarray = empty(0)
        self.bc_dof_ids = empty(0)
        self.field_variables: Dict[str, ndarray] = {}
        self.init_element_data_list()
        self.update_global_stiffness()
        # self.apply_bcs()

    def to_string(self, level: int = 1) -> str:
        return object_dict_to_string_assembly(self, level)

    def show(self) -> None:
        print(self.to_string())

    @show_running_time
    def init_element_data_list(self) -> None:
        # 初始化 self.element_data_list
        elements = self.props.elements
        nodes = self.props.nodes
        sections = self.props.sections
        materials = self.props.materials
        dof = self.props.dof
        dimension = nodes.dimension
        self.total_dof_number = len(nodes) * len(dof.names)

        for material in materials:
            self.materials_dict[material.name] = material

        for section in sections:
            self.sections_dict[section.name] = section

        for element_set in elements.element_sets:
            for section in sections:
                if element_set in section.element_sets:
                    self.section_of_element_set[element_set] = section

        for element_set_name, element_ids in elements.element_sets.items():
            section = self.section_of_element_set[element_set_name]
            material = self.materials_dict[section.material_name]
            material_data = get_material_data(material=material,
                                              dimension=dimension,
                                              option=section.type)

            for element_id in element_ids:
                connectivity = elements[element_id]
                node_coords = array(nodes.get_items_by_ids(list(connectivity)))
                iso_element_type = get_iso_element_type(node_coords)
                iso_element_shape = iso_element_shape_dict[iso_element_type]
                element_data = get_element_data(element_id=element_id,
                                                iso_element_shape=iso_element_shape,
                                                connectivity=connectivity,
                                                node_coords=node_coords,
                                                section=section,
                                                dof=dof,
                                                material=material,
                                                material_data=material_data)

                element_data.assembly_conn = array(nodes.get_indices_by_ids(list(connectivity)))
                element_data.create_element_dof_ids()

                self.element_data_list.append(element_data)

        # 初始化 self.bc_data_list
        bcs = self.props.bcs
        for bc in bcs:
            bc_data = get_bc_data(bc=bc, dof=dof, nodes=nodes)
            self.bc_data_list.append(bc_data)

        # 初始化 rhs, fext, fint, dof_solution
        self.rhs = zeros(self.total_dof_number)
        self.fext = zeros(self.total_dof_number)
        self.fint = zeros(self.total_dof_number)
        self.dof_solution = zeros(self.total_dof_number)

    @show_running_time
    def update_global_stiffness(self) -> None:
        self.update_element_data()

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

        self.global_stiffness = coo_matrix((array(val), (array(row), array(col))),
                                           shape=(self.total_dof_number, self.total_dof_number)).tocsc()

        # 以下代码采用 numpy.append 方法，处理可变对象时效率非常低，不建议使用

        # val = array([], dtype=float)
        # row = array([], dtype=int)
        # col = array([], dtype=int)
        #
        # for element_data in self.element_data_list:
        #     element_dof_ids = element_data.element_dof_ids
        #     element_dof_number = element_data.element_dof_number
        #     row = append(row, repeat(element_dof_ids, element_dof_number))
        #     for i in range(element_dof_number):
        #         col = append(col, element_dof_ids)
        #     val = append(val, element_data.stiffness.reshape(element_dof_number * element_dof_number))
        #
        # self.global_stiffness = coo_matrix((val, (row, col)), shape=(self.total_dof_number, self.total_dof_number))

    def apply_bcs(self) -> None:
        penalty = 1.0e16
        self.rhs = deepcopy(self.fext)
        bc_dof_ids = []
        for bc_data in self.bc_data_list:
            for dof_id, dof_value in zip(bc_data.dof_ids, bc_data.dof_values):
                bc_dof_ids.append(dof_id)
                self.global_stiffness[dof_id, dof_id] += penalty
                self.rhs[dof_id] += dof_value * penalty
        self.bc_dof_ids = array(bc_dof_ids)

    @show_running_time
    def update_fint(self) -> None:
        for element_data in self.element_data_list:
            element_fint = element_data.element_fint
            # print(element_data.connectivity)
            # print(element_data.element_dof_ids)
            # print(element_data.element_fint)
            # print(element_data.gp_jacobi_invs)
            # print(element_data.element_stiffness)
            element_dof_ids = element_data.element_dof_ids
            self.fint[element_dof_ids] += element_fint

    @show_running_time
    def update_element_data(self) -> None:
        solution = self.dof_solution
        for element_data in self.element_data_list:
            element_data.update_element_dof_values(solution)
            element_data.update_material_state()
            element_data.update_element_stiffness()
            element_data.update_element_fint()

    @show_running_time
    def update_field_variables(self) -> None:
        nodes_number = len(self.props.nodes)

        for element_data in self.element_data_list:
            element_data.update_element_field_variables()

        for output in self.props.outputs:
            if output.type == 'vtk':
                for field_name in output.field_outputs:
                    self.field_variables[field_name] = zeros(nodes_number)
                    nodes_count = zeros(nodes_number)
                    for element_data in self.element_data_list:
                        assembly_conn = element_data.assembly_conn
                        self.field_variables[field_name][assembly_conn] += element_data.average_field_variables[
                            field_name]
                        nodes_count[assembly_conn] += 1.0
                    self.field_variables[field_name] = self.field_variables[field_name] / nodes_count


@show_running_time
def main():
    from pyfem.solvers.get_solver_data import get_solver_data

    props = Properties()
    props.read_file(r'F:\Github\pyfem\examples\rectangle\rectangle.toml')
    props.verify()
    # props.show()
    assembly = Assembly(props)

    # x, info = gmres(A.toarray(), b, tol=1e-16)

    solver_data = get_solver_data(assembly, props.solver)

    solver_data.solve()

    solution = solver_data.solution

    assembly.update_field_variables(solution)

    import pprint
    pprint.pprint(assembly.field_variables)

    # print(props.nodes.node_sets['top'])


if __name__ == "__main__":
    main()
