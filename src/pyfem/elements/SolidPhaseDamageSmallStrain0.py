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


class SolidPhaseDamageSmallStrain(BaseElement):
    r"""
    固体相场断裂单元。

    :ivar qp_b_matrices: 积分点处的B矩阵列表
    :vartype qp_b_matrices: ndarray

    :ivar qp_b_matrices_transpose: 积分点处的B矩阵转置列表
    :vartype qp_b_matrices_transpose: ndarray

    :ivar qp_strains: 积分点处的应变列表
    :vartype qp_strains: list[ndarray]

    :ivar qp_stresses: 积分点处的应力列表
    :vartype qp_stresses: list[ndarray]

    :ivar qp_phases: 积分点处的相场变量列表
    :vartype qp_phases: list[ndarray]

    :ivar qp_phase_fluxes: 积分点处的相场变量通量列表
    :vartype qp_phase_fluxes: list[ndarray]

    :ivar qp_ddsddps: 积分点处的相场刚度矩阵列表
    :vartype qp_ddsddps: list[ndarray]

    :ivar dof_u: 单元位移自由度列表
    :vartype dof_u: list[int]

    :ivar dof_p: 单元相场自由度列表
    :vartype dof_p: list[int]

    :ivar ntens: 总应力数量
    :vartype ntens: int

    :ivar ndi: 轴向应力数量
    :vartype ndi: int

    :ivar nshr: 剪切应力数量
    :vartype nshr: int

    .. math::
        K_{ij}^\varphi  = \int_V {\left[ {{g_c}{l_c}{{\left( {\nabla {N_i}} \right)}^T}\nabla {N_j} + \left( {\frac{{{g_c}}}{{{l_c}}} + 2H} \right){N_i}{N_j}} \right]{\text{d}}V}

    .. math::
        RHS_i^\varphi  = \int_V {\left[ {{g_c}{l_c}{{\left( {\nabla {N_i}} \right)}^T}\nabla \varphi  - \left( {2\left( {1 - \varphi } \right)H - \frac{{{g_c}}}{{{l_c}}}\varphi } \right){N_i}} \right]{\text{d}}V}
    """

    __slots_dict__: dict = {
        'qp_b_matrices': ('ndarray', '积分点处的B矩阵列表'),
        'qp_b_matrices_transpose': ('ndarray', '积分点处的B矩阵转置列表'),
        'qp_strains': ('list[ndarray]', '积分点处的应变列表'),
        'qp_stresses': ('list[ndarray]', '积分点处的应力列表'),
        'qp_phases': ('list[ndarray]', '积分点处的相场变量列表'),
        'qp_phase_fluxes': ('list[ndarray]', '积分点处的相场变量通量列表'),
        'qp_ddsddps': ('list[ndarray]', '积分点处的相场刚度矩阵列表'),
        'qp_energies': ('list[ndarray]', '积分点处的相场刚度矩阵列表'),
        'dof_u': ('list[int]', '单元位移自由度列表'),
        'dof_p': ('list[int]', '单元相场自由度列表'),
        'ntens': ('int', '总应力数量'),
        'ndi': ('int', '轴向应力数量'),
        'nshr': ('int', '剪切应力数量')
    }

    __slots__ = BaseElement.__slots__ + [slot for slot in __slots_dict__.keys()]

    __allowed_material_data_list__ = [('ElasticIsotropic', 'PlasticKinematicHardening', 'ViscoElasticMaxwell', 'PlasticCrystal', 'PlasticCrystalGNDs', 'User'),
                                      ('PhaseFieldDamage', 'User')]

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
            self.dof_names = ['u1', 'u2', 'phi']
            self.ntens = 4
            self.ndi = 3
            self.nshr = 1
        elif self.dimension == 3:
            self.dof_names = ['u1', 'u2', 'u3', 'phi']
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
        self.qp_stresses: list[ndarray] = None  # type: ignore
        self.qp_phases: list[ndarray] = None  # type: ignore
        self.qp_phase_fluxes: list[ndarray] = None  # type: ignore
        self.qp_ddsddps: list[ndarray] = None  # type: ignore
        self.qp_energies: list[ndarray] = None  # type: ignore

        for i in range(self.qp_number):
            self.qp_state_variables[i]['history_energy'] = array([0.0])
            self.qp_state_variables_new[i]['history_energy'] = array([0.0])

        self.dof_u: list[int] = list()
        self.dof_p: list[int] = list()
        for i in range(self.iso_element_shape.nodes_number):
            if self.dimension == 2:
                self.dof_u += [len(self.dof_names) * i + 0, len(self.dof_names) * i + 1]
                self.dof_p += [len(self.dof_names) * i + 2]
            elif self.dimension == 3:
                self.dof_u += [len(self.dof_names) * i + 0, len(self.dof_names) * i + 1, len(self.dof_names) * i + 2]
                self.dof_p += [len(self.dof_names) * i + 3]
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

        dimension = self.iso_element_shape.dimension

        qp_number = self.qp_number
        qp_shape_values = self.iso_element_shape.qp_shape_values
        qp_shape_gradients = self.iso_element_shape.qp_shape_gradients
        qp_dhdxes = self.qp_dhdxes

        qp_b_matrices = self.qp_b_matrices
        qp_b_matrices_transpose = self.qp_b_matrices_transpose
        qp_jacobi_invs = self.qp_jacobi_invs
        qp_weight_times_jacobi_dets = self.qp_weight_times_jacobi_dets

        qp_state_variables = self.qp_state_variables
        qp_state_variables_new = self.qp_state_variables_new

        element_dof_values = self.element_dof_values
        element_ddof_values = self.element_ddof_values

        u = element_dof_values[self.dof_u]
        phi = element_dof_values[self.dof_p]

        du = element_ddof_values[self.dof_u]
        dphi = element_ddof_values[self.dof_p]

        solid_material_data = self.material_data_list[0]
        phase_material_data = self.material_data_list[1]

        gc = phase_material_data.gc  # type: ignore
        lc = phase_material_data.lc  # type: ignore

        if is_update_stiffness:
            self.element_stiffness = zeros(shape=(self.element_dof_number, self.element_dof_number), dtype=DTYPE)

        if is_update_fint:
            self.element_fint = zeros(self.element_dof_number, dtype=DTYPE)

        if is_update_material:
            self.qp_ddsddes = list()
            self.qp_strains = list()
            self.qp_stresses = list()
            self.qp_ddsddps = list()
            self.qp_phases = list()
            self.qp_phase_fluxes = list()
            self.qp_energies = list()

        for i in range(qp_number):
            if is_update_material:
                qp_weight_times_jacobi_det = qp_weight_times_jacobi_dets[i]
                qp_shape_value = qp_shape_values[i]
                qp_dhdx = qp_dhdxes[i]
                qp_b_matrix_transpose = qp_b_matrices_transpose[i]
                qp_b_matrix = qp_b_matrices[i]
                qp_strain = dot(qp_b_matrix, u)
                qp_dstrain = dot(qp_b_matrix, du)
                qp_phase = dot(qp_shape_value, phi)
                qp_dphase = dot(qp_shape_value, dphi)
                qp_phase_gradient = dot(qp_dhdx, phi)
                qp_dphase_gradient = dot(qp_dhdx, dphi)

                qp_degradation = (1.0 - qp_phase) ** 2 + 1.0e-8
                qp_degradation = min(qp_degradation, 1.0)
                qp_degradation = max(qp_degradation, 0.0)

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
                qp_strain_energy = qp_output['strain_energy']
                self.qp_ddsddes.append(qp_ddsdde)
                self.qp_strains.append(qp_strain)
                self.qp_stresses.append(qp_stress * qp_degradation)
                self.qp_phases.append(qp_phase)

            else:
                qp_b_matrix_transpose = qp_b_matrices_transpose[i]
                qp_b_matrix = qp_b_matrices[i]
                qp_weight_times_jacobi_det = qp_weight_times_jacobi_dets[i]
                qp_shape_value = qp_shape_values[i]
                qp_dhdx = qp_dhdxes[i]
                qp_ddsdde = self.qp_ddsddes[i]
                qp_stress = self.qp_stresses[i]
                qp_strain = dot(qp_b_matrix, u)
                qp_dstrain = dot(qp_b_matrix, du)
                qp_phase = dot(qp_shape_value, phi)
                qp_dphase = dot(qp_shape_value, dphi)
                qp_phase_gradient = dot(qp_dhdx, phi)
                qp_dphase_gradient = dot(qp_dhdx, dphi)

                qp_degradation = (1.0 - qp_phase) ** 2 + 1.0e-8
                qp_degradation = min(qp_degradation, 1.0)
                qp_degradation = max(qp_degradation, 0.0)

            # energy_positive, energy_negative = get_decompose_energy(qp_strain + qp_dstrain, qp_stress, dimension)
            # energy_positive += qp_strain_energy

            energy_positive = 0.5 * sum((qp_strain + qp_dstrain) * qp_stress)

            if energy_positive < qp_state_variables[i]['history_energy'][0]:
                energy_positive = qp_state_variables[i]['history_energy'][0]

            if energy_positive < qp_state_variables_new[i]['history_energy'][0]:
                energy_positive = qp_state_variables_new[i]['history_energy'][0]

            qp_state_variables_new[i]['history_energy'][0] = energy_positive

            self.qp_energies.append(energy_positive)

            if is_update_stiffness:
                self.element_stiffness[ix_(self.dof_u, self.dof_u)] += qp_weight_times_jacobi_det * \
                                                                       dot(qp_b_matrix_transpose, dot(qp_ddsdde * qp_degradation, qp_b_matrix))

                self.element_stiffness[ix_(self.dof_p, self.dof_p)] += qp_weight_times_jacobi_det * \
                                                                       ((gc / lc + 2.0 * energy_positive) * outer(qp_shape_value, qp_shape_value) +
                                                                        gc * lc * dot(qp_dhdx.transpose(), qp_dhdx))

                # vecu = -2.0 * (1.0 - (qp_phase + qp_dphase)) * dot(qp_b_matrix_transpose, qp_stress * qp_degradation) * qp_weight_times_jacobi_det
                # self.element_stiffness[ix_(self.dof_u, self.dof_p)] += outer(vecu, qp_shape_value)
                # self.element_stiffness[ix_(self.dof_p, self.dof_u)] += outer(qp_shape_value, vecu)

            if is_update_fint:
                self.element_fint[self.dof_u] += dot(qp_b_matrix_transpose, qp_stress * qp_degradation) * qp_weight_times_jacobi_det

                self.element_fint[self.dof_p] += qp_weight_times_jacobi_det * \
                                                 (gc * lc * dot(qp_dhdx.transpose(), (qp_phase_gradient + qp_dphase_gradient)) +
                                                  gc / lc * (qp_phase + qp_dphase) * qp_shape_value +
                                                  2.0 * ((qp_phase + qp_dphase) - 1.0) * energy_positive * qp_shape_value)

    def update_element_field_variables(self) -> None:
        qp_stresses = self.qp_stresses
        qp_strains = self.qp_strains
        qp_dstrains = self.qp_dstrains
        qp_energies = self.qp_energies

        average_strain = average(qp_strains, axis=0) + average(qp_dstrains, axis=0)
        average_stress = average(qp_stresses, axis=0)
        average_energy = average(qp_energies, axis=0)

        self.qp_field_variables['strain'] = array(qp_strains, dtype=DTYPE)
        self.qp_field_variables['stress'] = array(qp_stresses, dtype=DTYPE)

        if self.dimension == 2:
            self.element_nodal_field_variables['E11'] = average_strain[0]
            self.element_nodal_field_variables['E22'] = average_strain[1]
            self.element_nodal_field_variables['E12'] = average_strain[2]
            self.element_nodal_field_variables['S11'] = average_stress[0]
            self.element_nodal_field_variables['S22'] = average_stress[1]
            self.element_nodal_field_variables['S12'] = average_stress[2]
            self.element_nodal_field_variables['ENERGY'] = average_energy

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
            self.element_nodal_field_variables['ENERGY'] = average_energy


if __name__ == "__main__":
    from pyfem.utils.visualization import print_slots_dict

    print_slots_dict(SolidPhaseDamageSmallStrain.__slots_dict__)

    from pyfem.Job import Job

    job = Job(r'..\..\..\examples\mechanical_phase\rectangle\Job-1.toml')

    job.assembly.element_data_list[0].show()
