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
from pyfem.materials.get_material_data import MaterialData
from pyfem.utils.colors import error_style
from pyfem.utils.mechanics import get_decompose_energy


class SolidPhaseFieldDamagePlaneSmallStrain(BaseElement):
    __slots__ = BaseElement.__slots__ + ['gp_b_matrices',
                                         'gp_b_matrices_transpose',
                                         'gp_strains',
                                         'gp_stresses',
                                         'gp_phases',
                                         'gp_phase_fluxes',
                                         'gp_ddsddps',
                                         'dof_u',
                                         'dof_p']

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
                                           ('PhaseFieldDamage',)]
        self.allowed_material_number = len(self.allowed_material_data_list)

        self.dof = dof
        self.materials = materials
        self.section = section
        self.material_data_list = material_data_list
        self.check_materials()
        self.timer = timer

        self.dof_names = ['u1', 'u2', 'phi']
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
        self.gp_strains: List[ndarray] = None  # type: ignore
        self.gp_stresses: List[ndarray] = None  # type: ignore
        self.gp_phases: List[ndarray] = None  # type: ignore
        self.gp_phase_fluxes: List[ndarray] = None  # type: ignore
        self.gp_ddsddps: List[ndarray] = None  # type: ignore

        # for i in range(self.gp_number):
        #     self.gp_state_variables[i]['history_energy'] = array([0.0])
        #     self.gp_state_variables_new[i]['history_energy'] = array([0.0])

        self.dof_u = []
        self.dof_p = []
        for i in range(self.iso_element_shape.nodes_number):
            self.dof_u += [len(self.dof_names) * i + 0, len(self.dof_names) * i + 1]
            self.dof_p += [len(self.dof_names) * i + 2]

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

        dimension = self.iso_element_shape.dimension

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
        phi = element_dof_values[self.dof_p]

        du = element_ddof_values[self.dof_u]
        dphi = element_ddof_values[self.dof_p]

        self.element_stiffness = zeros(shape=(self.element_dof_number, self.element_dof_number), dtype=DTYPE)
        self.element_fint = zeros(self.element_dof_number, dtype=DTYPE)

        solid_material_data = self.material_data_list[0]
        phase_material_data = self.material_data_list[1]

        gc = phase_material_data.gc  # type: ignore
        lc = phase_material_data.lc  # type: ignore

        gp_ddsddes = []
        gp_strains = []
        gp_stresses = []
        # gp_ddsddps = []
        gp_phases = []
        # gp_phase_fluxes = []

        for i in range(gp_number):
            gp_weight_times_jacobi_det = gp_weight_times_jacobi_dets[i]
            gp_shape_value = gp_shape_values[i]
            gp_shape_gradient = gp_shape_gradients[i]
            gp_b_matrix_transpose = gp_b_matrices_transpose[i]
            gp_b_matrix = gp_b_matrices[i]
            gp_strain = dot(gp_b_matrix, u)
            gp_dstrain = dot(gp_b_matrix, du)
            gp_phase = dot(gp_shape_value, phi)
            gp_dphase = dot(gp_shape_value, dphi)
            gp_phase_gradient = dot(gp_shape_gradient, phi)
            gp_dphase_gradient = dot(gp_shape_gradient, dphi)

            degradation = min((1.0 - gp_phase) ** 2 + 1.0e-7, 1.0)

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
            gp_stress = gp_output['stress'] * degradation

            energy_positive, energy_negative = get_decompose_energy(gp_strain + gp_dstrain, gp_stress, dimension)

            # if energy_positive > gp_state_variables_new[i]['history_energy'][0]:
            #     gp_state_variables_new[i]['history_energy'][0] = energy_positive
            # else:
            #     energy_positive = gp_state_variables_new[i]['history_energy'][0]

            self.element_stiffness[ix_(self.dof_u, self.dof_u)] += \
                dot(gp_b_matrix_transpose, dot(gp_ddsdde, gp_b_matrix)) * gp_weight_times_jacobi_det * degradation

            self.element_fint[self.dof_u] += dot(gp_b_matrix_transpose, gp_stress) * gp_weight_times_jacobi_det

            self.element_stiffness[ix_(self.dof_p, self.dof_p)] += \
                ((gc / lc + 2.0 * energy_positive) * outer(gp_shape_value, gp_shape_value) +
                 gc * lc * dot((gp_phase + gp_dphase), (gp_phase + gp_dphase).transpose())) * gp_weight_times_jacobi_det

            self.element_fint[self.dof_p] += \
                (gc * lc * dot(gp_shape_gradient.transpose(), gp_phase_gradient) +
                 gc / lc * gp_shape_value * (gp_phase + gp_dphase) +
                 2.0 * ((gp_phase + gp_dphase) - 1.0) * gp_shape_value * energy_positive) * gp_weight_times_jacobi_det

            gp_ddsddes.append(gp_ddsdde)
            gp_strains.append(gp_strain)
            gp_stresses.append(gp_stress)
            # gp_ddsddps.append(gp_ddsddp)
            gp_phases.append(gp_phase)
            # gp_phase_fluxes.append(gp_phase_flux)

        self.gp_ddsddes = gp_ddsddes
        self.gp_strains = gp_strains
        self.gp_stresses = gp_stresses
        # self.gp_ddsddps = gp_ddsddps
        self.gp_phases = gp_phases
        # self.gp_phase_fluxes = gp_phase_fluxes

    def update_element_stiffness(self) -> None:
        raise NotImplementedError()

    def update_element_fint(self) -> None:
        raise NotImplementedError()

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
    pass
