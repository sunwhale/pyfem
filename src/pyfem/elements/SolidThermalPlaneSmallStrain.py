# -*- coding: utf-8 -*-
"""

"""
from typing import List

from numpy import array, zeros, dot, ndarray, average, ix_, outer

from pyfem.elements.BaseElement import BaseElement
from pyfem.elements.IsoElementShape import IsoElementShape
from pyfem.fem.Timer import Timer
from pyfem.fem.constants import DTYPE
from pyfem.io.Dof import Dof
from pyfem.io.Material import Material
from pyfem.io.Section import Section
from pyfem.utils.colors import error_style
from pyfem.utils.data_types import MaterialData


class SolidThermalPlaneSmallStrain(BaseElement):
    __slots__ = BaseElement.__slots__ + ['gp_b_matrices', 'gp_b_matrices_transpose', 'gp_strains', 'gp_stresses',
                                         'gp_temperatures', 'gp_heat_fluxes', 'gp_ddsddts', 'dof_u', 'dof_T']

    def __init__(self, element_id: int,
                 iso_element_shape: IsoElementShape,
                 connectivity: ndarray,
                 node_coords: ndarray,
                 dof: Dof,
                 materials: List[Material],
                 section: Section,
                 material_data_list: List[MaterialData],
                 timer: Timer) -> None:

        super().__init__(element_id, iso_element_shape, connectivity, node_coords)

        self.allowed_material_data_list = [('ElasticIsotropic', 'PlasticKinematicHardening', 'ViscoElasticMaxwell'),
                                           ('ThermalIsotropic',),
                                           ('MechanicalThermalExpansion',)]
        self.allowed_material_number = len(self.allowed_material_data_list)

        self.dof = dof
        self.materials = materials
        self.section = section
        self.material_data_list = material_data_list
        self.check_materials()
        self.timer = timer

        self.dof_names = ['u1', 'u2', 'T']
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
        self.gp_strains: List[ndarray] = []
        self.gp_stresses: List[ndarray] = []
        self.gp_heat_fluxes: List[ndarray] = None  # type: ignore
        self.gp_temperatures: List[ndarray] = None  # type: ignore
        self.gp_ddsddts: List[ndarray] = []

        self.dof_u = []
        self.dof_T = []
        for i in range(self.iso_element_shape.nodes_number):
            self.dof_u += [len(self.dof_names) * i + 0, len(self.dof_names) * i + 1]
            self.dof_T += [len(self.dof_names) * i + 2]

        self.create_gp_b_matrices()

    def create_gp_b_matrices(self) -> None:
        self.gp_b_matrices = zeros(shape=(self.gp_number, 3, len(self.dof_u)), dtype=DTYPE)

        for igp, (gp_shape_gradient, gp_jacobi_inv) in \
                enumerate(zip(self.iso_element_shape.gp_shape_gradients, self.gp_jacobi_invs)):
            gp_dhdx = dot(gp_shape_gradient.transpose(), gp_jacobi_inv)
            for i, val in enumerate(gp_dhdx):
                self.gp_b_matrices[igp, 0, i * 2] = val[0]
                self.gp_b_matrices[igp, 1, i * 2 + 1] = val[1]
                self.gp_b_matrices[igp, 2, i * 2] = val[1]
                self.gp_b_matrices[igp, 2, i * 2 + 1] = val[0]

        self.gp_b_matrices_transpose = array([gp_b_matrix.transpose() for gp_b_matrix in self.gp_b_matrices])

    def update_material_state(self) -> None:
        pass

    def update_element_material_stiffness_fint(self) -> None:
        element_id = self.element_id
        timer = self.timer

        gp_number = self.gp_number
        gp_shape_values = self.iso_element_shape.gp_shape_values
        gp_shape_gradients = self.iso_element_shape.gp_shape_gradients
        gp_b_matrices = self.gp_b_matrices
        gp_b_matrices_transpose = self.gp_b_matrices_transpose
        gp_weight_times_jacobi_dets = self.gp_weight_times_jacobi_dets

        gp_state_variables = self.gp_state_variables
        gp_state_variables_new = self.gp_state_variables_new

        element_dof_values = self.element_dof_values
        element_ddof_values = self.element_ddof_values

        u = element_dof_values[self.dof_u]
        T = element_dof_values[self.dof_T]

        du = element_ddof_values[self.dof_u]
        dT = element_ddof_values[self.dof_T]

        self.element_stiffness = zeros(shape=(self.element_dof_number, self.element_dof_number), dtype=DTYPE)
        self.element_fint = zeros(self.element_dof_number, dtype=DTYPE)

        solid_material_data = self.material_data_list[0]
        thermal_material_data = self.material_data_list[1]
        solid_thermal_material_data = self.material_data_list[2]

        alpha = solid_thermal_material_data.tangent

        gp_ddsddes = []
        gp_strains = []
        gp_stresses = []

        for i in range(gp_number):
            gp_weight_times_jacobi_det = gp_weight_times_jacobi_dets[i]
            gp_shape_value = gp_shape_values[i]
            gp_b_matrix_transpose = gp_b_matrices_transpose[i]
            gp_b_matrix = gp_b_matrices[i]
            gp_strain = dot(gp_b_matrix, u)
            gp_dstrain = dot(gp_b_matrix, du)
            gp_temperature = dot(gp_shape_value, T)
            gp_dtemperature = dot(gp_shape_value, dT)
            gp_strain += (-alpha * gp_temperature)
            gp_dstrain += (-alpha * gp_dtemperature)
            variable = {'strain': gp_strain, 'dstrain': gp_dstrain}
            gp_ddsdde, gp_output = solid_material_data.get_tangent(variable=variable,
                                                                   state_variable=gp_state_variables[i],
                                                                   state_variable_new=gp_state_variables_new[i],
                                                                   element_id=element_id,
                                                                   igp=i,
                                                                   ntens=4,
                                                                   ndi=3,
                                                                   nshr=1,
                                                                   timer=timer)
            gp_stress = gp_output['stress']

            self.element_stiffness[ix_(self.dof_u, self.dof_u)] += \
                dot(gp_b_matrix_transpose, dot(gp_ddsdde, gp_b_matrix)) * gp_weight_times_jacobi_det

            dsdt = -1.0 * dot(gp_ddsdde, alpha)
            self.element_stiffness[ix_(self.dof_u, self.dof_T)] += \
                dot(gp_b_matrix_transpose, outer(dsdt, gp_shape_value)) * gp_weight_times_jacobi_det

            self.element_fint[self.dof_u] += dot(gp_b_matrix_transpose, gp_stress) * gp_weight_times_jacobi_det

            gp_ddsddes.append(gp_ddsdde)
            gp_strains.append(gp_strain)
            gp_stresses.append(gp_stress)

        self.gp_ddsddes = gp_ddsddes
        self.gp_strains = gp_strains
        self.gp_stresses = gp_stresses

        gp_ddsddts = []
        gp_temperatures = []
        gp_heat_fluxes = []

        for i in range(gp_number):
            gp_weight_times_jacobi_det = gp_weight_times_jacobi_dets[i]
            gp_shape_value = gp_shape_values[i]
            gp_shape_gradient = gp_shape_gradients[i]
            gp_temperature = dot(gp_shape_value, T)
            gp_dtemperature = dot(gp_shape_value, dT)
            gp_temperature_gradient = dot(gp_shape_gradient, T)
            gp_dtemperature_gradient = dot(gp_shape_gradient, dT)

            variable = {'temperature': gp_temperature,
                        'dtemperature': gp_dtemperature,
                        'temperature_gradient': gp_temperature_gradient,
                        'dtemperature_gradient': gp_dtemperature_gradient}
            gp_ddsddt, gp_output = thermal_material_data.get_tangent(variable=variable,
                                                                     state_variable=gp_state_variables[i],
                                                                     state_variable_new=gp_state_variables_new[i],
                                                                     element_id=element_id,
                                                                     igp=i,
                                                                     ntens=4,
                                                                     ndi=3,
                                                                     nshr=1,
                                                                     timer=timer)
            gp_heat_flux = gp_output['heat_flux']

            self.element_stiffness[ix_(self.dof_T, self.dof_T)] += \
                dot(gp_shape_gradient.transpose(), dot(gp_ddsddt, gp_shape_gradient)) * gp_weight_times_jacobi_det

            self.element_fint[self.dof_T] += \
                dot(gp_shape_gradient.transpose(), gp_heat_flux) * gp_weight_times_jacobi_det

            gp_ddsddts.append(gp_ddsddt)
            gp_temperatures.append(gp_temperature)
            gp_heat_fluxes.append(gp_heat_flux)

        self.gp_ddsddts = gp_ddsddts
        self.gp_temperatures = gp_temperatures
        self.gp_heat_fluxes = gp_heat_fluxes

    def update_element_stiffness(self) -> None:
        self.element_stiffness = zeros(shape=(self.element_dof_number, self.element_dof_number), dtype=DTYPE)

        gp_weight_times_jacobi_dets = self.gp_weight_times_jacobi_dets
        gp_shape_gradients = self.iso_element_shape.gp_shape_gradients
        gp_b_matrices = self.gp_b_matrices
        gp_b_matrices_transpose = self.gp_b_matrices_transpose
        gp_number = self.gp_number
        gp_ddsddes = self.gp_ddsddes
        gp_ddsddts = self.gp_ddsddts

        for i in range(gp_number):
            self.element_stiffness[ix_(self.dof_u, self.dof_u)] += \
                dot(gp_b_matrices_transpose[i], dot(gp_ddsddes[i], gp_b_matrices[i])) * gp_weight_times_jacobi_dets[i]

            self.element_stiffness[ix_(self.dof_T, self.dof_T)] += \
                dot(gp_shape_gradients[i].transpose(), dot(gp_ddsddts[i], gp_shape_gradients[i])) * \
                gp_weight_times_jacobi_dets[i]

    def update_element_fint(self) -> None:
        gp_weight_times_jacobi_dets = self.gp_weight_times_jacobi_dets
        gp_shape_gradients = self.iso_element_shape.gp_shape_gradients
        gp_b_matrices_transpose = self.gp_b_matrices_transpose
        gp_number = self.gp_number
        gp_stresses = self.gp_stresses
        gp_heat_fluxes = self.gp_heat_fluxes

        self.element_fint = zeros(self.element_dof_number, dtype=DTYPE)
        for i in range(gp_number):
            self.element_fint[self.dof_u] += dot(gp_b_matrices_transpose[i], gp_stresses[i]) * \
                                             gp_weight_times_jacobi_dets[i]

            self.element_fint[self.dof_T] += dot(gp_shape_gradients[i].transpose(), gp_heat_fluxes[i]) * \
                                             gp_weight_times_jacobi_dets[i]

    def update_element_field_variables(self) -> None:
        gp_stresses = self.gp_stresses
        gp_strains = self.gp_strains

        average_strain = average(gp_strains, axis=0)
        average_stress = average(gp_stresses, axis=0)

        self.gp_field_variables['strain'] = array(gp_strains, dtype=DTYPE)
        self.gp_field_variables['stress'] = array(gp_stresses, dtype=DTYPE)

        self.element_average_field_variables['E11'] = average_strain[0]
        self.element_average_field_variables['E22'] = average_strain[1]
        self.element_average_field_variables['E12'] = average_strain[2]
        self.element_average_field_variables['S11'] = average_stress[0]
        self.element_average_field_variables['S22'] = average_stress[1]
        self.element_average_field_variables['S12'] = average_stress[2]


if __name__ == "__main__":
    from pyfem.Job import Job

    job = Job(r'F:\Github\pyfem\examples\1element\hex8\Job-1.toml')

    job.assembly.element_data_list[0].show()
