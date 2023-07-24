# -*- coding: utf-8 -*-
"""

"""
from numpy import array, zeros, dot, ndarray, average

from pyfem.elements.BaseElement import BaseElement
from pyfem.fem.Timer import Timer
from pyfem.fem.constants import DTYPE
from pyfem.io.Dof import Dof
from pyfem.io.Material import Material
from pyfem.io.Section import Section
from pyfem.isoelements.IsoElementShape import IsoElementShape
from pyfem.materials.get_material_data import MaterialData
from pyfem.utils.colors import error_style


class SolidSmallStrain(BaseElement):
    """
    固体小变形单元。

    :ivar gp_b_matrices: 积分点处的B矩阵列表
    :vartype gp_b_matrices: ndarray

    :ivar gp_b_matrices_transpose: 积分点处的B矩阵转置列表
    :vartype gp_b_matrices_transpose: ndarray

    :ivar gp_strains: 积分点处的应变列表
    :vartype gp_strains: list[ndarray]

    :ivar gp_stresses: 积分点处的应力列表
    :vartype gp_stresses: list[ndarray]

    :ivar ntens: 总应力数量
    :vartype ntens: int

    :ivar ndi: 轴向应力数量
    :vartype ndi: int

    :ivar nshr: 剪切应力数量
    :vartype nshr: int
    """

    __slots_dict__: dict = {
        'gp_b_matrices': ('ndarray', '积分点处的B矩阵列表'),
        'gp_b_matrices_transpose': ('ndarray', '积分点处的B矩阵转置列表'),
        'gp_strains': ('list[ndarray]', '积分点处的应变列表'),
        'gp_stresses': ('list[ndarray]', '积分点处的应力列表'),
        'ntens': ('int', '总应力数量'),
        'ndi': ('int', '轴向应力数量'),
        'nshr': ('int', '剪切应力数量')
    }

    __slots__: list = BaseElement.__slots__ + [slot for slot in __slots_dict__.keys()]

    def __init__(self, element_id: int,
                 iso_element_shape: IsoElementShape,
                 connectivity: ndarray,
                 node_coords: ndarray,
                 dof: Dof,
                 materials: list[Material],
                 section: Section,
                 material_data_list: list[MaterialData],
                 timer: Timer) -> None:

        super().__init__(element_id, iso_element_shape, connectivity, node_coords)

        self.allowed_material_data_list = [('ElasticIsotropic', 'PlasticKinematicHardening', 'PlasticCrystal', 'ViscoElasticMaxwell')]
        self.allowed_material_number = 1

        self.dof = dof
        self.materials = materials
        self.section = section
        self.material_data_list = material_data_list
        self.check_materials()
        self.timer = timer

        if self.dimension == 2:
            self.dof_names = ['u1', 'u2']
            self.ntens = 4
            self.ndi = 3
            self.nshr = 1
        elif self.dimension == 3:
            self.dof_names = ['u1', 'u2', 'u3']
            self.ntens = 6
            self.ndi = 3
            self.nshr = 3
        else:
            error_msg = f'{self.dimension} is not the supported dimension'
            raise NotImplementedError(error_style(error_msg))

        if dof.names != self.dof_names:
            error_msg = f'{dof.names} is not the supported dof of {type(self).__name__} element'
            raise NotImplementedError(error_style(error_msg))

        element_dof_number = len(self.dof_names) * self.iso_element_shape.nodes_number
        self.element_dof_number = element_dof_number
        self.element_dof_values = zeros(element_dof_number, dtype=DTYPE)
        self.element_ddof_values = zeros(element_dof_number, dtype=DTYPE)
        self.element_fint = zeros(element_dof_number, dtype=DTYPE)
        self.element_stiffness = None  # type: ignore

        self.gp_b_matrices: ndarray = None  # type: ignore
        self.gp_b_matrices_transpose: ndarray = None  # type: ignore
        self.gp_strains: list[ndarray] = None  # type: ignore
        self.gp_stresses: list[ndarray] = None  # type: ignore

        self.create_gp_b_matrices()

    def create_gp_b_matrices(self) -> None:
        if self.dimension == 2:
            self.gp_b_matrices = zeros(shape=(self.gp_number, 3, self.element_dof_number), dtype=DTYPE)
            for igp, (gp_shape_gradient, gp_jacobi_inv) in \
                    enumerate(zip(self.iso_element_shape.gp_shape_gradients, self.gp_jacobi_invs)):
                gp_dhdx = dot(gp_shape_gradient.transpose(), gp_jacobi_inv)
                for i, val in enumerate(gp_dhdx):
                    self.gp_b_matrices[igp, 0, i * 2] = val[0]
                    self.gp_b_matrices[igp, 1, i * 2 + 1] = val[1]
                    self.gp_b_matrices[igp, 2, i * 2] = val[1]
                    self.gp_b_matrices[igp, 2, i * 2 + 1] = val[0]

        elif self.dimension == 3:
            self.gp_b_matrices = zeros(shape=(self.iso_element_shape.gp_number, 6, self.element_dof_number))
            for igp, (gp_shape_gradient, gp_jacobi_inv) in \
                    enumerate(zip(self.iso_element_shape.gp_shape_gradients, self.gp_jacobi_invs)):
                gp_dhdx = dot(gp_shape_gradient.transpose(), gp_jacobi_inv)
                for i, val in enumerate(gp_dhdx):
                    self.gp_b_matrices[igp, 0, i * 3] = val[0]
                    self.gp_b_matrices[igp, 1, i * 3 + 1] = val[1]
                    self.gp_b_matrices[igp, 2, i * 3 + 2] = val[2]
                    self.gp_b_matrices[igp, 3, i * 3] = val[1]
                    self.gp_b_matrices[igp, 3, i * 3 + 1] = val[0]
                    self.gp_b_matrices[igp, 4, i * 3] = val[2]
                    self.gp_b_matrices[igp, 4, i * 3 + 2] = val[0]
                    self.gp_b_matrices[igp, 5, i * 3 + 1] = val[2]
                    self.gp_b_matrices[igp, 5, i * 3 + 2] = val[1]

        self.gp_b_matrices_transpose = array([gp_b_matrix.transpose() for gp_b_matrix in self.gp_b_matrices])

    def update_element_material_stiffness_fint(self,
                                               is_update_material: bool = True,
                                               is_update_stiffness: bool = True,
                                               is_update_fint: bool = True, ) -> None:
        element_id = self.element_id
        timer = self.timer
        ntens = self.ntens
        ndi = self.ndi
        nshr = self.nshr

        gp_number = self.gp_number
        gp_b_matrices = self.gp_b_matrices
        gp_b_matrices_transpose = self.gp_b_matrices_transpose
        gp_weight_times_jacobi_dets = self.gp_weight_times_jacobi_dets

        gp_state_variables = self.gp_state_variables
        gp_state_variables_new = self.gp_state_variables_new

        element_dof_values = self.element_dof_values
        element_ddof_values = self.element_ddof_values

        material_data = self.material_data_list[0]

        if is_update_stiffness:
            self.element_stiffness = zeros(shape=(self.element_dof_number, self.element_dof_number), dtype=DTYPE)

        if is_update_fint:
            self.element_fint = zeros(self.element_dof_number, dtype=DTYPE)

        if is_update_material:
            self.gp_ddsddes = list()
            self.gp_strains = list()
            self.gp_stresses = list()

        for i in range(gp_number):
            if is_update_material:
                gp_weight_times_jacobi_det = gp_weight_times_jacobi_dets[i]
                gp_b_matrix_transpose = gp_b_matrices_transpose[i]
                gp_b_matrix = gp_b_matrices[i]
                gp_strain = dot(gp_b_matrix, element_dof_values)
                gp_dstrain = dot(gp_b_matrix, element_ddof_values)
                variable = {'strain': gp_strain, 'dstrain': gp_dstrain}
                gp_ddsdde, gp_output = material_data.get_tangent(variable=variable,
                                                                 state_variable=gp_state_variables[i],
                                                                 state_variable_new=gp_state_variables_new[i],
                                                                 element_id=element_id,
                                                                 igp=i,
                                                                 ntens=ntens,
                                                                 ndi=ndi,
                                                                 nshr=nshr,
                                                                 timer=timer)
                gp_stress = gp_output['stress']
                self.gp_ddsddes.append(gp_ddsdde)
                self.gp_strains.append(gp_strain)
                self.gp_stresses.append(gp_stress)
            else:
                gp_b_matrix_transpose = gp_b_matrices_transpose[i]
                gp_b_matrix = gp_b_matrices[i]
                gp_weight_times_jacobi_det = gp_weight_times_jacobi_dets[i]
                gp_ddsdde = self.gp_ddsddes[i]
                gp_stress = self.gp_stresses[i]

            if is_update_stiffness:
                self.element_stiffness += dot(gp_b_matrix_transpose, dot(gp_ddsdde, gp_b_matrix)) * \
                                          gp_weight_times_jacobi_det

            if is_update_fint:
                self.element_fint += dot(gp_b_matrix_transpose, gp_stress) * gp_weight_times_jacobi_det

    def update_element_field_variables(self) -> None:
        gp_stresses = self.gp_stresses
        gp_strains = self.gp_strains

        average_strain = average(gp_strains, axis=0)
        average_stress = average(gp_stresses, axis=0)

        self.gp_field_variables['strain'] = array(gp_strains, dtype=DTYPE)
        self.gp_field_variables['stress'] = array(gp_stresses, dtype=DTYPE)

        if self.dimension == 2:
            self.element_average_field_variables['E11'] = average_strain[0]
            self.element_average_field_variables['E22'] = average_strain[1]
            self.element_average_field_variables['E12'] = average_strain[2]
            self.element_average_field_variables['S11'] = average_stress[0]
            self.element_average_field_variables['S22'] = average_stress[1]
            self.element_average_field_variables['S12'] = average_stress[2]

        elif self.dimension == 3:
            self.element_average_field_variables['E11'] = average_strain[0]
            self.element_average_field_variables['E22'] = average_strain[1]
            self.element_average_field_variables['E33'] = average_strain[2]
            self.element_average_field_variables['E12'] = average_strain[3]
            self.element_average_field_variables['E13'] = average_strain[4]
            self.element_average_field_variables['E23'] = average_strain[5]
            self.element_average_field_variables['S11'] = average_stress[0]
            self.element_average_field_variables['S22'] = average_stress[1]
            self.element_average_field_variables['S33'] = average_stress[2]
            self.element_average_field_variables['S12'] = average_stress[3]
            self.element_average_field_variables['S13'] = average_stress[4]
            self.element_average_field_variables['S23'] = average_stress[5]


if __name__ == "__main__":
    from pyfem.utils.visualization import print_slots_dict

    print_slots_dict(SolidSmallStrain.__slots_dict__)

    # from pyfem.Job import Job
    #
    # job = Job(r'..\..\..\examples\mechanical\1element\hex8\Job-1.toml')
    #
    # job.assembly.element_data_list[0].show()
