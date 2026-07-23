# -*- coding: utf-8 -*-
"""

"""
import numpy as np

from pyfem.elements.BaseElement import BaseElement
from pyfem.elements.set_element_field_variables import set_element_field_variables
from pyfem.fem.Timer import Timer
from pyfem.fem.constants import DTYPE
from pyfem.io.Dof import Dof
from pyfem.io.Material import Material
from pyfem.io.Section import Section
from pyfem.isoelements.IsoElementShape import IsoElementShape
from pyfem.materials.get_material_data import MaterialData
from pyfem.utils.colors import error_style


class SolidPhaseDamageFengSmallStrain(BaseElement):
    r"""
    固体相场断裂单元。

    :ivar qp_b_matrices: 积分点处的B矩阵列表
    :vartype qp_b_matrices: np.ndarray

    :ivar qp_b_matrices_transpose: 积分点处的B矩阵转置列表
    :vartype qp_b_matrices_transpose: np.ndarray

    :ivar qp_strains: 积分点处的应变列表
    :vartype qp_strains: list[np.ndarray]

    :ivar qp_stresses: 积分点处的应力列表
    :vartype qp_stresses: list[np.ndarray]

    :ivar qp_phases: 积分点处的相场变量列表
    :vartype qp_phases: list[np.ndarray]

    :ivar qp_phase_fluxes: 积分点处的相场变量通量列表
    :vartype qp_phase_fluxes: list[np.ndarray]

    :ivar qp_ddsddps: 积分点处的相场刚度矩阵列表
    :vartype qp_ddsddps: list[np.ndarray]

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
        'qp_b_matrices': ('np.ndarray', '积分点处的B矩阵列表'),
        'qp_b_matrices_transpose': ('np.ndarray', '积分点处的B矩阵转置列表'),
        'qp_strains': ('list[np.ndarray]', '积分点处的应变列表'),
        'qp_dstrains': ('list[np.ndarray]', '积分点处的应变增量列表'),
        'qp_stresses': ('list[np.ndarray]', '积分点处的应力列表'),
        'qp_phases': ('list[np.ndarray]', '积分点处的相场变量列表'),
        'qp_phase_fluxes': ('list[np.ndarray]', '积分点处的相场变量通量列表'),
        'qp_ddsddps': ('list[np.ndarray]', '积分点处的相场刚度矩阵列表'),
        'qp_energies': ('list[np.ndarray]', '积分点处的应变能列表'),
        'qp_damages': ('list[np.ndarray]', '积分点处的损伤参数列表'),
        'dof_u': ('list[int]', '单元位移自由度列表'),
        'dof_p': ('list[int]', '单元相场自由度列表'),
        'ntens': ('int', '总应力数量'),
        'ndi': ('int', '轴向应力数量'),
        'nshr': ('int', '剪切应力数量')
    }

    __slots__ = BaseElement.__slots__ + [slot for slot in __slots_dict__.keys()]

    __allowed_material_data_list__ = [('ElasticIsotropic', 'PlasticKinematicHardening', 'ViscoElasticMaxwell', 'PlasticCrystal', 'PlasticCrystalGNDs', 'User'),
                                      ('PhaseFieldDamageFeng', 'User')]

    def __init__(self, element_id: int,
                 iso_element_shape: IsoElementShape,
                 connectivity: np.ndarray,
                 node_coords: np.ndarray,
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
        self.element_dof_values = np.zeros(element_dof_number, dtype=DTYPE)
        self.element_ddof_values = np.zeros(element_dof_number, dtype=DTYPE)
        self.element_fint = np.zeros(element_dof_number, dtype=DTYPE)
        self.element_stiffness = np.zeros(shape=(self.element_dof_number, self.element_dof_number), dtype=DTYPE)

        self.qp_b_matrices: np.ndarray = None  # type: ignore
        self.qp_b_matrices_transpose: np.ndarray = None  # type: ignore
        self.qp_strains: list[np.ndarray] = None  # type: ignore
        self.qp_dstrains: list[np.ndarray] = None  # type: ignore
        self.qp_stresses: list[np.ndarray] = None  # type: ignore
        self.qp_phases: list[np.ndarray] = None  # type: ignore
        self.qp_phase_fluxes: list[np.ndarray] = None  # type: ignore
        self.qp_ddsddps: list[np.ndarray] = None  # type: ignore
        self.qp_energies: list[np.ndarray] = None  # type: ignore
        self.qp_damages: list[np.ndarray] = None  # type: ignore

        for i in range(self.qp_number):
            self.qp_state_variables[i]['history_energy'] = np.array([0.0])
            self.qp_state_variables_new[i]['history_energy'] = np.array([0.0])
            self.qp_state_variables[i]['damage'] = np.array([0.0])
            self.qp_state_variables_new[i]['damage'] = np.array([0.0])

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
            self.qp_b_matrices = np.zeros(shape=(self.qp_number, 3, len(self.dof_u)), dtype=DTYPE)
            for iqp, qp_dhdx in enumerate(self.qp_dhdxes):
                for i, val in enumerate(qp_dhdx.transpose()):
                    self.qp_b_matrices[iqp, 0, i * 2 + 0] = val[0]
                    self.qp_b_matrices[iqp, 1, i * 2 + 1] = val[1]
                    self.qp_b_matrices[iqp, 2, i * 2 + 0] = val[1]
                    self.qp_b_matrices[iqp, 2, i * 2 + 1] = val[0]

        elif self.dimension == 3:
            self.qp_b_matrices = np.zeros(shape=(self.iso_element_shape.qp_number, 6, len(self.dof_u)), dtype=DTYPE)
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

        self.qp_b_matrices_transpose = np.array([qp_b_matrix.transpose() for qp_b_matrix in self.qp_b_matrices])

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
        a1 = phase_material_data.a1  # type: ignore
        a2 = phase_material_data.a2  # type: ignore
        a3 = phase_material_data.a3  # type: ignore
        p = phase_material_data.p  # type: ignore
        xi = phase_material_data.xi  # type: ignore
        c0 = phase_material_data.c0  # type: ignore
        gth = phase_material_data.gth  # type: ignore
        # print(gc, lc, a1, a2, a3, p, xi, c0)

        ft = 200.0
        E = 1.0e5

        if is_update_stiffness:
            self.element_stiffness = np.zeros(shape=(self.element_dof_number, self.element_dof_number), dtype=DTYPE)

        if is_update_fint:
            self.element_fint = np.zeros(self.element_dof_number, dtype=DTYPE)

        if is_update_material:
            self.qp_ddsddes = list()
            self.qp_strains = list()
            self.qp_dstrains = list()
            self.qp_stresses = list()
            self.qp_ddsddps = list()
            self.qp_phases = list()
            self.qp_phase_fluxes = list()
            self.qp_energies = list()
            self.qp_damages = list()

        for i in range(qp_number):
            if is_update_material:
                qp_weight_times_jacobi_det = qp_weight_times_jacobi_dets[i]
                qp_shape_value = qp_shape_values[i]
                qp_dhdx = qp_dhdxes[i]
                qp_b_matrix_transpose = qp_b_matrices_transpose[i]
                qp_b_matrix = qp_b_matrices[i]
                qp_strain = np.dot(qp_b_matrix, u)
                qp_dstrain = np.dot(qp_b_matrix, du)
                qp_phase = np.dot(qp_shape_value, phi)
                qp_dphase = np.dot(qp_shape_value, dphi)
                qp_phase_gradient = np.dot(qp_dhdx, phi)
                qp_dphase_gradient = np.dot(qp_dhdx, dphi)

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

                qp_alpha, qp_dalpha, qp_ddalpha = geometric_func(qp_phase + qp_dphase, xi)
                qp_omega, qp_domega, qp_ddomega = energetic_func(qp_phase + qp_dphase, a1, a2, a3, p)

                qp_energy = 0.5 * sum((qp_strain + qp_dstrain) * qp_stress) - gth

                if qp_energy < qp_state_variables[i]['history_energy'][0]:
                    qp_energy = qp_state_variables[i]['history_energy'][0]

                if qp_energy < qp_state_variables_new[i]['history_energy'][0]:
                    qp_energy = qp_state_variables_new[i]['history_energy'][0]

                qp_energy += 1.0e-8

                qp_state_variables_new[i]['history_energy'][0] = qp_energy

                w0 = (ft ** 2) / (2.0 * E)
                c1 = gc / (lc * w0)
                qp_damage = 0.0

                if qp_energy > w0:
                    qp_damage = 1.0 / c1 * ((np.sqrt(qp_energy / (1.0 - qp_phase) ** 2.0) / w0) - 1.0)

                qp_damage = max(qp_damage, 0.0)
                qp_omega = 1.0 / (1.0 + c1 * qp_damage)

                qp_state_variables_new[i]['damage'][0] = qp_damage
                qp_omega += 1.0e-8
                qp_omega = min(qp_omega, 1.0)
                qp_omega = max(qp_omega, 0.0)

                self.qp_ddsddes.append(qp_ddsdde)
                self.qp_strains.append(qp_strain)
                self.qp_dstrains.append(qp_dstrain)
                self.qp_stresses.append(qp_stress * qp_omega)
                self.qp_phases.append(qp_phase)
                self.qp_damages.append(qp_damage)

            else:
                qp_b_matrix_transpose = qp_b_matrices_transpose[i]
                qp_b_matrix = qp_b_matrices[i]
                qp_weight_times_jacobi_det = qp_weight_times_jacobi_dets[i]
                qp_shape_value = qp_shape_values[i]
                qp_dhdx = qp_dhdxes[i]
                qp_ddsdde = self.qp_ddsddes[i]
                qp_stress = self.qp_stresses[i]
                qp_strain = np.dot(qp_b_matrix, u)
                qp_dstrain = np.dot(qp_b_matrix, du)
                qp_phase = np.dot(qp_shape_value, phi)
                qp_dphase = np.dot(qp_shape_value, dphi)
                qp_phase_gradient = np.dot(qp_dhdx, phi)
                qp_dphase_gradient = np.dot(qp_dhdx, dphi)

                qp_alpha, qp_dalpha, qp_ddalpha = geometric_func(qp_phase + qp_dphase, xi)
                qp_omega, qp_domega, qp_ddomega = energetic_func(qp_phase + qp_dphase, a1, a2, a3, p)

                qp_energy = 0.5 * sum((qp_strain + qp_dstrain) * qp_stress) - gth

                if qp_energy < qp_state_variables[i]['history_energy'][0]:
                    qp_energy = qp_state_variables[i]['history_energy'][0]

                if qp_energy < qp_state_variables_new[i]['history_energy'][0]:
                    qp_energy = qp_state_variables_new[i]['history_energy'][0]

                qp_energy += 1.0e-8

                qp_state_variables_new[i]['history_energy'][0] = qp_energy

                w0 = (ft ** 2) / (2.0 * E)
                c1 = gc / (lc * w0)
                qp_damage = 0.0
                if qp_energy > w0:
                    qp_damage = 1.0 / c1 * ((np.sqrt(qp_energy / (1.0 - qp_phase) ** 2.0) / w0) - 1.0)

                qp_damage = max(qp_damage, 0.0)
                qp_omega = 1.0 / (1.0 + c1 * qp_damage)
                qp_state_variables_new[i]['damage'][0] = qp_damage

                # qp_omega = (1.0 - qp_phase) ** 2

                qp_omega += 1.0e-8
                qp_omega = min(qp_omega, 1.0)
                qp_omega = max(qp_omega, 0.0)

            self.qp_energies.append(qp_energy)
            self.qp_damages.append(qp_damage)

            if is_update_stiffness:
                self.element_stiffness[np.ix_(self.dof_u, self.dof_u)] += qp_weight_times_jacobi_det * \
                                                                          np.dot(qp_b_matrix_transpose, np.dot(qp_ddsdde * qp_omega, qp_b_matrix))

                self.element_stiffness[np.ix_(self.dof_p, self.dof_p)] += qp_weight_times_jacobi_det * \
                                                                          ((gc / lc + 2.0 * qp_energy) * np.outer(qp_shape_value, qp_shape_value) +
                                                                           gc * lc * np.dot(qp_dhdx.transpose(), qp_dhdx))

                # vecu = -2.0 * (1.0 - (qp_phase + qp_dphase)) * np.dot(qp_b_matrix_transpose, qp_stress * qp_omega) * qp_weight_times_jacobi_det
                # self.element_stiffness[np.ix_(self.dof_u, self.dof_p)] += np.outer(vecu, qp_shape_value)
                # self.element_stiffness[np.ix_(self.dof_p, self.dof_u)] += np.outer(qp_shape_value, vecu)

            if is_update_fint:
                self.element_fint[self.dof_u] += np.dot(qp_b_matrix_transpose, qp_stress * qp_omega) * qp_weight_times_jacobi_det

                self.element_fint[self.dof_p] += qp_weight_times_jacobi_det * \
                                                 (gc * lc * np.dot(qp_dhdx.transpose(), (qp_phase_gradient + qp_dphase_gradient)) +
                                                  gc / lc * (qp_phase + qp_dphase) * qp_shape_value +
                                                  2.0 * ((qp_phase + qp_dphase) - 1.0) * qp_energy * qp_shape_value)

    def update_element_field_variables(self) -> None:
        self.qp_field_variables['strain'] = np.array(self.qp_strains, dtype=DTYPE) + np.array(self.qp_dstrains, dtype=DTYPE)
        self.qp_field_variables['stress'] = np.array(self.qp_stresses, dtype=DTYPE)
        self.qp_field_variables['energy'] = np.array(self.qp_energies, dtype=DTYPE)
        self.qp_field_variables['damage'] = np.array(self.qp_damages, dtype=DTYPE)
        for key in self.qp_state_variables_new[0].keys():
            if key not in ['strain', 'stress', 'energy']:
                variable = []
                for qp_state_variable_new in self.qp_state_variables_new:
                    variable.append(qp_state_variable_new[key])
                self.qp_field_variables[f'SDV-{key}'] = np.array(variable, dtype=DTYPE)
        self.element_nodal_field_variables = set_element_field_variables(self.qp_field_variables, self.iso_element_shape, self.dimension, self.nodes_number)


def geometric_func(phi, xi):
    alpha = xi * phi + (1.0 - xi) * phi ** 2.0
    dalpha = xi + 2.0 * (1.0 - xi) * phi
    ddalpha = 2.0 * (1.0 - xi)

    return alpha, dalpha, ddalpha


def energetic_func(phi, a1, a2, a3, p):
    fac1 = (1.0 - phi) ** p
    dfac1 = -p * (1.0 - phi) ** (p - 1.0)
    ddfac1 = p * (p - 1.0) * (1.0 - phi) ** (p - 2.0)

    fac2 = fac1 + a1 * phi + a1 * a2 * phi ** 2.0 + a1 * a2 * a3 * phi ** 3.0
    dfac2 = dfac1 + a1 + 2.0 * a1 * a2 * phi + 3.0 * a1 * a2 * a3 * phi ** 2.0
    ddfac2 = ddfac1 + 2.0 * a1 * a2 + 6.0 * a1 * a2 * a3 * phi

    omega = fac1 / fac2
    domega = (dfac1 * fac2 - fac1 * dfac2) / (fac2 ** 2.0)
    ddomega = ((ddfac1 * fac2 - fac1 * ddfac2) * fac2 - 2.0 * (dfac1 * fac2 - fac1 * dfac2) * dfac2) / (fac2 ** 3.0)

    return omega, domega, ddomega


def stress_to_tensor(stress):
    stress_tensor = np.array([[stress[0], stress[2]], [stress[2], stress[1]]])
    return stress_tensor


def principle_stress_2d(stress):
    stress_tensor = stress_to_tensor(stress)

    s11 = stress_tensor[0, 0]
    s12 = stress_tensor[0, 1]
    s22 = stress_tensor[1, 1]

    s1 = (s11 + s22) / 2 + np.sqrt(((s11 - s22) / 2) ** 2 + s12 ** 2)
    s2 = (s11 + s22) / 2 - np.sqrt(((s11 - s22) / 2) ** 2 + s12 ** 2)

    tol = 1e-14

    if (s1 - s2) ** 2 < tol:
        # ---------- 内层条件：判断剪应力是否可忽略 ----------
        if s12 ** 2 < tol:
            # 纯静水压力状态，角度无定义，取 0
            theta0 = 0.0
        else:
            # 纯剪切状态，角度为 ±45°
            if s12 > 0.0:
                theta0 = np.pi / 4.0
            else:
                theta0 = -np.pi / 4.0
    else:
        # ---------- 一般情况（单轴或一般双轴状态） ----------
        # 使用 atan2(b, a) 代替 atan(b/a) 以安全处理 a = 0 的情形
        a = s11 - s22
        b = 2.0 * s12

        theta0 = 0.5 * np.atan2(b, a)

        if a < 0.0 and b < 0.0:
            theta0 += np.pi

    return s1, s2, theta0


def sigma_rotate(sigma, theta):
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    sigma_R = R.T @ sigma @ R
    return sigma_R, R


def positive_part(x):
    """正部函数 ⟨x⟩ = max(x, 0)"""
    return np.maximum(x, 0.0)


def heaviside(x):
    """Heaviside 阶跃函数，x>0 时返回 1，否则 0"""
    return 1.0 if x > 0.0 else 0.0


def sigma_degenerate(sigma, phi, theta, c1, c2):
    """
    计算总应力（损伤模型）
    """
    # 旋转到局部坐标系
    sigma_R, R = sigma_rotate(sigma, theta)
    sigma_R_11, sigma_R_22, sigma_R_12 = sigma_R[0, 0], sigma_R[1, 1], sigma_R[0, 1]

    # 正部
    sigma_R_11_positive = positive_part(sigma_R_11)
    sigma_R_22_positive = positive_part(sigma_R_22)

    # 局部应力分量
    sigma_R_11_degenerate = sigma_R_11 - (1.0 - g(phi, c1)) * sigma_R_11_positive
    sigma_R_22_degenerate = sigma_R_22 - (1.0 - g(phi, c1)) * sigma_R_22_positive
    sigma_R_12_degenerate = g(phi, c2) * sigma_R_12

    # 局部应力张量
    sigma_R_degenerate = np.array([[sigma_R_11_degenerate, sigma_R_12_degenerate],
                                   [sigma_R_12_degenerate, sigma_R_22_degenerate]])

    # 转换回全局坐标
    sigma_degenerate = R @ sigma_R_degenerate @ R.T
    return sigma_degenerate


def g(phi, coef):
    return 1 / (1 + coef * phi) + 1e-5


def dg(phi, coef):
    return -coef / (1 + coef * phi) ** 2


def H_D(S_local, D, theta, ft, tau_cr, c1, c2, g1, g2):
    """
    计算损伤驱动量 H0
    参数：
        S_local : 2x2 numpy array，局部坐标系下的弹性应力张量
        D       : 损伤变量（标量，仅用于函数签名，实际计算中可不使用）
        theta   : 裂缝角度（弧度），此处仅用于说明，实际局部应力已给出
        ft      : 抗拉强度
        tau_cr  : 临界剪切应力
        c1, c2  : 权重参数
        g1, g2  : 损伤退化函数值 g1(D), g2(D)
    返回：
        H0      : 标量
    """
    S_nn = S_local[0, 0]
    S_nm = S_local[0, 1]  # 或 S_local[1,0]

    # 正部
    pos_Snn = positive_part(S_nn)

    # 根据条件选择公式
    if c1 >= c2:
        # 拉伸主导
        term1 = (pos_Snn / ft) ** 2
        term2 = ((g2 / g1) ** 2) * (S_nm / tau_cr) ** 2
    else:
        # 剪切主导
        term1 = ((g1 / g2) ** 2) * (pos_Snn / ft) ** 2
        term2 = (S_nm / tau_cr) ** 2

    H0 = term1 + term2
    return H0


if __name__ == "__main__":
    from pyfem.utils.visualization import print_slots_dict

    # print_slots_dict(SolidPhaseDamageFengSmallStrain.__slots_dict__)
    #
    # c0 = 2.0
    # lc = 1.0
    # gc = 3.96826441
    # E = 1.0
    # ft = 1.0
    # a1 = 4.575058194
    # a2 = 0.0
    # a3 = 0.0
    # p = 2.0
    # xi = 0.0
    # print(geometric_func(0.12218250952777067, xi))
    # print(energetic_func(0.12218250952777067, a1, a2, a3, p))

    n_dimension = 2
    cohesive_law = 'E'
    E = 3e4
    nu = 0.2
    E1 = E / (1 - nu ** 2)
    ft = 3
    Gc = 0.04
    l0 = 5
    lmbda = nu * E / (1 - nu ** 2)  # 2d
    mu = E / 2 / (1 + nu)
    K = lmbda + 2 * mu / n_dimension
    W_cr = ft ** 2 / 2 / E1
    tau_cr = ft * 5
    beta = ft / tau_cr
    W1 = ft ** 2 / 2 / E1
    W2 = tau_cr ** 2 / 2 / mu
    c2 = Gc / W2 / l0
    c1 = Gc / W1 / l0

    S = np.array([[10.0, 2.0],
                  [2.0, 5.0]])  # 单位 MPa

    D = 0.1
    theta = np.radians(30)
    nu = 0.3

    # 总应力
    sigma = sigma_degenerate(S, D, theta, c1, c2)
    print("Total stress:\n", sigma)
