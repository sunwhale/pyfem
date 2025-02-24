# -*- coding: utf-8 -*-
"""

"""
from numpy import array, zeros, dot, ndarray, average, ix_, outer

from pyfem.elements.BaseElement import BaseElement
from pyfem.fem.Timer import Timer
from pyfem.fem.constants import DTYPE
from pyfem.io.Dof import Dof
from pyfem.io.Material import Material
from pyfem.io.Section import Section
from pyfem.isoelements.IsoElementShape import IsoElementShape
from pyfem.materials.get_material_data import MaterialData
from pyfem.utils.colors import error_style


class SolidThermalSmallStrain(BaseElement):
    """
    固体变形-温度场耦合单元。

    :ivar qp_b_matrices: 积分点处的B矩阵列表
    :vartype qp_b_matrices: ndarray

    :ivar qp_b_matrices_transpose: 积分点处的B矩阵转置列表
    :vartype qp_b_matrices_transpose: ndarray

    :ivar qp_strains: 积分点处的应变列表
    :vartype qp_strains: list[ndarray]

    :ivar qp_stresses: 积分点处的应力列表
    :vartype qp_stresses: list[ndarray]

    :ivar ntens: 总应力数量
    :vartype ntens: int

    :ivar ndi: 轴向应力数量
    :vartype ndi: int

    :ivar nshr: 剪切应力数量
    :vartype nshr: int
    """

    __slots_dict__: dict = {
        'qp_b_matrices': ('ndarray', '积分点处的B矩阵列表'),
        'qp_b_matrices_transpose': ('ndarray', '积分点处的B矩阵转置列表'),
        'qp_strains': ('list[ndarray]', '积分点处的应变列表'),
        'qp_dstrains': ('list[ndarray]', '积分点处的应变增量列表'),
        'qp_stresses': ('list[ndarray]', '积分点处的应力列表'),
        'qp_temperatures': ('ndarray', '积分点处的温度列表'),
        'qp_heat_fluxes': ('ndarray', '积分点处的热流密度列表'),
        'qp_ddsddts': ('list[ndarray]', '积分点处的材料热传导系数矩阵列表'),
        'dof_u': ('list[int]', '单元位移自由度列表'),
        'dof_T': ('list[int]', '单元温度自由度列表'),
        'ntens': ('int', '总应力数量'),
        'ndi': ('int', '轴向应力数量'),
        'nshr': ('int', '剪切应力数量')
    }

    __slots__: list = BaseElement.__slots__ + [slot for slot in __slots_dict__.keys()]

    __allowed_material_data_list__ = [('ElasticIsotropic', 'PlasticKinematicHardening', 'ViscoElasticMaxwell', 'User'),
                                      ('ThermalIsotropic', 'User'),
                                      ('MechanicalThermalExpansion', 'User')]

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

        self.allowed_material_data_list = self.__allowed_material_data_list__
        self.allowed_material_number = len(self.allowed_material_data_list)

        self.dof = dof
        self.materials = materials
        self.section = section
        self.material_data_list = material_data_list
        self.check_materials()
        self.timer = timer

        if self.dimension == 2:
            self.dof_names = ['u1', 'u2', 'T']
            self.ntens = 4
            self.ndi = 3
            self.nshr = 1
        elif self.dimension == 3:
            self.dof_names = ['u1', 'u2', 'u3', 'T']
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
        self.element_stiffness = zeros(shape=(self.element_dof_number, self.element_dof_number), dtype=DTYPE)

        self.qp_b_matrices: ndarray = None  # type: ignore
        self.qp_b_matrices_transpose: ndarray = None  # type: ignore
        self.qp_strains: list[ndarray] = None  # type: ignore
        self.qp_dstrains: list[ndarray] = None  # type: ignore
        self.qp_stresses: list[ndarray] = None  # type: ignore
        self.qp_temperatures: list[ndarray] = None  # type: ignore
        self.qp_heat_fluxes: list[ndarray] = None  # type: ignore
        self.qp_ddsddts: list[ndarray] = None  # type: ignore

        self.dof_u: list[int] = list()
        self.dof_T: list[int] = list()
        for i in range(self.iso_element_shape.nodes_number):
            self.dof_u += [len(self.dof_names) * i + 0, len(self.dof_names) * i + 1]
            self.dof_T += [len(self.dof_names) * i + 2]

        self.create_qp_b_matrices()

    def create_qp_b_matrices(self) -> None:
        if self.dimension == 2:
            self.qp_b_matrices = zeros(shape=(self.qp_number, 3, len(self.dof_u)), dtype=DTYPE)
            for iqp, qp_dhdx in enumerate(self.qp_dhdxes):
                for i, val in enumerate(qp_dhdx.transpose()):
                    self.qp_b_matrices[iqp, 0, i * 2 + 0] = val[0]
                    self.qp_b_matrices[iqp, 1, i * 2 + 1] = val[1]
                    self.qp_b_matrices[iqp, 2, i * 2 + 0] = val[1]
                    self.qp_b_matrices[iqp, 2, i * 2 + 1] = val[0]

        elif self.dimension == 3:
            self.qp_b_matrices = zeros(shape=(self.iso_element_shape.qp_number, 6, len(self.dof_u)), dtype=DTYPE)
            for iqp, qp_dhdx in enumerate(self.qp_dhdxes):
                for i, val in enumerate(qp_dhdx.transpose()):
                    self.qp_b_matrices[iqp, 0, i * 3 + 0] = val[0]
                    self.qp_b_matrices[iqp, 1, i * 3 + 1] = val[1]
                    self.qp_b_matrices[iqp, 2, i * 3 + 2] = val[2]
                    self.qp_b_matrices[iqp, 3, i * 3 + 0] = val[1]
                    self.qp_b_matrices[iqp, 3, i * 3 + 1] = val[0]
                    self.qp_b_matrices[iqp, 4, i * 3 + 0] = val[2]
                    self.qp_b_matrices[iqp, 4, i * 3 + 2] = val[0]
                    self.qp_b_matrices[iqp, 5, i * 3 + 1] = val[2]
                    self.qp_b_matrices[iqp, 5, i * 3 + 2] = val[1]

        self.qp_b_matrices_transpose = array([qp_b_matrix.transpose() for qp_b_matrix in self.qp_b_matrices])

    def update_element_material_stiffness_fint(self,
                                               is_update_material: bool = True,
                                               is_update_stiffness: bool = True,
                                               is_update_fint: bool = True, ) -> None:
        element_id = self.element_id
        timer = self.timer
        ntens = self.ntens
        ndi = self.ndi
        nshr = self.nshr

        qp_number = self.qp_number
        qp_shape_values = self.iso_element_shape.qp_shape_values
        qp_dhdxes = self.qp_dhdxes
        qp_b_matrices = self.qp_b_matrices
        qp_b_matrices_transpose = self.qp_b_matrices_transpose
        qp_weight_times_jacobi_dets = self.qp_weight_times_jacobi_dets

        qp_state_variables = self.qp_state_variables
        qp_state_variables_new = self.qp_state_variables_new

        element_dof_values = self.element_dof_values
        element_ddof_values = self.element_ddof_values

        u = element_dof_values[self.dof_u]
        T = element_dof_values[self.dof_T]

        du = element_ddof_values[self.dof_u]
        dT = element_ddof_values[self.dof_T]

        solid_material_data = self.material_data_list[0]
        thermal_material_data = self.material_data_list[1]
        solid_thermal_material_data = self.material_data_list[2]

        alpha = solid_thermal_material_data.tangent

        if is_update_stiffness:
            self.element_stiffness = zeros(shape=(self.element_dof_number, self.element_dof_number), dtype=DTYPE)

        if is_update_fint:
            self.element_fint = zeros(self.element_dof_number, dtype=DTYPE)

        if is_update_material:
            self.qp_ddsddes = list()
            self.qp_strains = list()
            self.qp_dstrains = list()
            self.qp_stresses = list()

        for i in range(qp_number):
            if is_update_material:
                qp_weight_times_jacobi_det = qp_weight_times_jacobi_dets[i]
                qp_shape_value = qp_shape_values[i]
                qp_b_matrix_transpose = qp_b_matrices_transpose[i]
                qp_b_matrix = qp_b_matrices[i]
                qp_strain = dot(qp_b_matrix, u)
                qp_dstrain = dot(qp_b_matrix, du)
                qp_temperature = dot(qp_shape_value, T)
                qp_dtemperature = dot(qp_shape_value, dT)
                qp_strain += (-alpha * qp_temperature)
                qp_dstrain += (-alpha * qp_dtemperature)
                variable = {'strain': qp_strain, 'dstrain': qp_dstrain}
                qp_ddsdde, qp_output = solid_material_data.get_tangent(variable=variable,
                                                                       state_variable=qp_state_variables[i],
                                                                       state_variable_new=qp_state_variables_new[i],
                                                                       element_id=element_id,
                                                                       iqp=i,
                                                                       ntens=ntens,
                                                                       ndi=ndi,
                                                                       nshr=nshr,
                                                                       timer=timer)
                qp_stress = qp_output['stress']
                self.qp_ddsddes.append(qp_ddsdde)
                self.qp_strains.append(qp_strain)
                self.qp_dstrains.append(qp_dstrain)
                self.qp_stresses.append(qp_stress)
            else:
                qp_b_matrix_transpose = qp_b_matrices_transpose[i]
                qp_b_matrix = qp_b_matrices[i]
                qp_weight_times_jacobi_det = qp_weight_times_jacobi_dets[i]
                qp_shape_value = qp_shape_values[i]
                qp_ddsdde = self.qp_ddsddes[i]
                qp_stress = self.qp_stresses[i]

            if is_update_stiffness:
                self.element_stiffness[ix_(self.dof_u, self.dof_u)] += \
                    dot(qp_b_matrix_transpose, dot(qp_ddsdde, qp_b_matrix)) * qp_weight_times_jacobi_det

            if is_update_fint:
                dsdt = -1.0 * dot(qp_ddsdde, alpha)
                self.element_stiffness[ix_(self.dof_u, self.dof_T)] += \
                    dot(qp_b_matrix_transpose, outer(dsdt, qp_shape_value)) * qp_weight_times_jacobi_det

                self.element_fint[self.dof_u] += dot(qp_b_matrix_transpose, qp_stress) * qp_weight_times_jacobi_det

        if is_update_material:
            self.qp_ddsddts = list()
            self.qp_temperatures = list()
            self.qp_heat_fluxes = list()

        for i in range(qp_number):
            if is_update_material:
                qp_weight_times_jacobi_det = qp_weight_times_jacobi_dets[i]
                qp_shape_value = qp_shape_values[i]
                qp_dhdx = qp_dhdxes[i]
                qp_temperature = dot(qp_shape_value, T)
                qp_dtemperature = dot(qp_shape_value, dT)
                qp_temperature_gradient = dot(qp_dhdx, T)
                qp_dtemperature_gradient = dot(qp_dhdx, dT)

                variable = {'temperature': qp_temperature,
                            'dtemperature': qp_dtemperature,
                            'temperature_gradient': qp_temperature_gradient,
                            'dtemperature_gradient': qp_dtemperature_gradient}
                qp_ddsddt, qp_output = thermal_material_data.get_tangent(variable=variable,
                                                                         state_variable=qp_state_variables[i],
                                                                         state_variable_new=qp_state_variables_new[i],
                                                                         element_id=element_id,
                                                                         iqp=i,
                                                                         ntens=ntens,
                                                                         ndi=ndi,
                                                                         nshr=nshr,
                                                                         timer=timer)
                qp_heat_flux = qp_output['heat_flux']
                self.qp_ddsddts.append(qp_ddsddt)
                self.qp_temperatures.append(qp_temperature)
                self.qp_heat_fluxes.append(qp_heat_flux)
            else:
                qp_weight_times_jacobi_det = qp_weight_times_jacobi_dets[i]
                qp_dhdx = qp_dhdxes[i]
                qp_ddsddt = self.qp_ddsddts[i]
                qp_heat_flux = self.qp_heat_fluxes[i]

            if is_update_stiffness:
                self.element_stiffness[ix_(self.dof_T, self.dof_T)] += \
                    dot(qp_dhdx.transpose(), dot(qp_ddsddt, qp_dhdx)) * qp_weight_times_jacobi_det

            if is_update_fint:
                self.element_fint[self.dof_T] += \
                    dot(qp_dhdx.transpose(), qp_heat_flux) * qp_weight_times_jacobi_det

    def update_element_field_variables(self) -> None:
        qp_stresses = self.qp_stresses
        qp_strains = self.qp_strains
        qp_dstrains = self.qp_dstrains

        average_strain = average(qp_strains, axis=0) + average(qp_dstrains, axis=0)
        average_stress = average(qp_stresses, axis=0)

        self.qp_field_variables['strain'] = array(qp_strains, dtype=DTYPE)
        self.qp_field_variables['stress'] = array(qp_stresses, dtype=DTYPE)

        if self.dimension == 2:
            self.element_nodal_field_variables['E11'] = average_strain[0]
            self.element_nodal_field_variables['E22'] = average_strain[1]
            self.element_nodal_field_variables['E12'] = average_strain[2]
            self.element_nodal_field_variables['S11'] = average_stress[0]
            self.element_nodal_field_variables['S22'] = average_stress[1]
            self.element_nodal_field_variables['S12'] = average_stress[2]

        elif self.dimension == 3:
            self.element_nodal_field_variables['E11'] = average_strain[0]
            self.element_nodal_field_variables['E22'] = average_strain[1]
            self.element_nodal_field_variables['E33'] = average_strain[2]
            self.element_nodal_field_variables['E12'] = average_strain[3]
            self.element_nodal_field_variables['E13'] = average_strain[4]
            self.element_nodal_field_variables['E23'] = average_strain[5]
            self.element_nodal_field_variables['S11'] = average_stress[0]
            self.element_nodal_field_variables['S22'] = average_stress[1]
            self.element_nodal_field_variables['S33'] = average_stress[2]
            self.element_nodal_field_variables['S12'] = average_stress[3]
            self.element_nodal_field_variables['S13'] = average_stress[4]
            self.element_nodal_field_variables['S23'] = average_stress[5]


if __name__ == "__main__":
    from pyfem.utils.visualization import print_slots_dict

    print_slots_dict(SolidThermalSmallStrain.__slots_dict__)

    # from pyfem.Job import Job
    #
    # job = Job(r'..\..\..\examples\mechanical_thermal\rectangle\Job-1.toml')
    #
    # job.assembly.element_data_list[0].show()
