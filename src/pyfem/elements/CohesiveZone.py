# -*- coding: utf-8 -*-
"""

"""
from copy import deepcopy

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


class CohesiveZone(BaseElement):
    """
    **内聚力单元**

    :ivar qp_b_matrices: 积分点处的B矩阵列表
    :vartype qp_b_matrices: np.ndarray

    :ivar qp_b_matrices_transpose: 积分点处的B矩阵转置列表
    :vartype qp_b_matrices_transpose: np.ndarray

    :ivar qp_strains: 积分点处的应变列表
    :vartype qp_strains: list[np.ndarray]

    :ivar qp_stresses: 积分点处的应力列表
    :vartype qp_stresses: list[np.ndarray]

    :ivar ntens: 总应力数量
    :vartype ntens: int

    :ivar ndi: 轴向应力数量
    :vartype ndi: int

    :ivar nshr: 剪切应力数量
    :vartype nshr: int
    """

    __slots_dict__: dict = {
        'qp_b_matrices': ('np.ndarray', '积分点处的B矩阵列表'),
        'qp_b_matrices_transpose': ('np.ndarray', '积分点处的B矩阵转置列表'),
        'qp_strains': ('list[np.ndarray]', '积分点处的应变列表'),
        'qp_dstrains': ('list[np.ndarray]', '积分点处的应变增量列表'),
        'qp_stresses': ('list[np.ndarray]', '积分点处的应力列表'),
        'ntens': ('int', '总应力数量'),
        'ndi': ('int', '轴向应力数量'),
        'nshr': ('int', '剪切应力数量'),
        'normal': ('np.ndarray', '单元表面法向量'),
    }

    __slots__: list = BaseElement.__slots__ + [slot for slot in __slots_dict__.keys()]

    __allowed_material_data_list__ = [('Cohesive', 'User')]

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
        self.allowed_material_number = 1

        self.dof = dof
        self.materials = materials
        self.section = section
        self.material_data_list = material_data_list
        self.check_materials()
        self.timer = timer

        if self.dimension == 2:
            self.dof_names = ['u1', 'u2']
            self.ntens = 2
            self.ndi = 1
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

        element_dof_number = len(self.dof_names) * self.nodes_number
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

        self.create_normal()
        self.create_qp_b_matrices()

    def create_normal(self) -> None:
        if self.dimension == 2:
            node_coords = np.pad(self.node_coords, pad_width=((0, 0), (0, 1)), mode='constant', constant_values=0)
            v1 = node_coords[1, :] - node_coords[0, :]
            v2 = np.array([0.0, 0.0, 1.0])
        elif self.dimension == 3:
            node_coords = self.node_coords
            v1 = node_coords[2, :] - node_coords[0, :]
            v2 = node_coords[1, :] - node_coords[0, :]
        else:
            error_msg = f'{self.dimension} is not the supported dimension'
            raise ValueError(error_style(error_msg))

        self.normal = np.cross(v1, v2)
        normal_length = np.linalg.norm(self.normal)
        if normal_length > 1e-16:
            self.normal /= normal_length
        else:
            raise ValueError(error_style('The three points are collinear and cannot compute the normal vector.'))

        self.normal = self.normal[:self.dimension]

    def create_qp_b_matrices(self) -> None:
        if self.dimension == 2:
            node_coords = np.pad(self.node_coords, pad_width=((0, 0), (0, 1)), mode='constant', constant_values=0)
            mid_coords = deepcopy(node_coords[:2, :])

            mid_coords[0, 0] += 0.5 * (self.element_dof_values[0] + self.element_dof_values[6])
            mid_coords[0, 1] += 0.5 * (self.element_dof_values[1] + self.element_dof_values[7])
            mid_coords[1, 0] += 0.5 * (self.element_dof_values[2] + self.element_dof_values[4])
            mid_coords[1, 1] += 0.5 * (self.element_dof_values[3] + self.element_dof_values[5])

            ds = mid_coords[1, :] - mid_coords[0, :]
            z_axis = np.array([0.0, 0.0, 1.0])

            normal = np.cross(ds, z_axis)
            normal_length = np.linalg.norm(normal)
            if normal_length > 1e-16:
                normal /= normal_length
            else:
                raise ValueError(error_style('The three points are collinear and cannot compute the normal vector.'))

            normal = normal[:self.dimension]

            rot = np.array([[normal[0], normal[1]],
                            [normal[1], -normal[0]]])

            self.qp_b_matrices = np.zeros(shape=(self.qp_number, 2, self.element_dof_number), dtype=DTYPE)
            for iqp, _ in enumerate(self.qp_jacobis):
                self.qp_b_matrices[iqp, :, :2] = -rot * self.iso_element_shape.qp_shape_values[iqp, 0]
                self.qp_b_matrices[iqp, :, 2:4] = -rot * self.iso_element_shape.qp_shape_values[iqp, 1]
                self.qp_b_matrices[iqp, :, 4:6] = rot * self.iso_element_shape.qp_shape_values[iqp, 0]
                self.qp_b_matrices[iqp, :, 6:] = rot * self.iso_element_shape.qp_shape_values[iqp, 1]

        elif self.dimension == 3:
            self.qp_b_matrices = np.zeros(shape=(self.iso_element_shape.qp_number, 6, self.element_dof_number), dtype=DTYPE)

        self.qp_b_matrices_transpose = np.array([qp_b_matrix.transpose() for qp_b_matrix in self.qp_b_matrices])

    def update_element_material_stiffness_fint(self,
                                               is_update_material: bool = True,
                                               is_update_stiffness: bool = True,
                                               is_update_fint: bool = True, ) -> None:

        self.create_qp_b_matrices()

        element_id = self.element_id
        timer = self.timer
        ntens = self.ntens
        ndi = self.ndi
        nshr = self.nshr

        qp_number = self.qp_number
        qp_b_matrices = self.qp_b_matrices
        qp_b_matrices_transpose = self.qp_b_matrices_transpose
        qp_weight_times_jacobi_dets = self.qp_weight_times_jacobi_dets

        qp_state_variables = self.qp_state_variables
        qp_state_variables_new = self.qp_state_variables_new

        element_dof_values = self.element_dof_values
        element_ddof_values = self.element_ddof_values

        material_data = self.material_data_list[0]

        if is_update_stiffness:
            self.element_stiffness = np.zeros(shape=(self.element_dof_number, self.element_dof_number), dtype=DTYPE)

        if is_update_fint:
            self.element_fint = np.zeros(self.element_dof_number, dtype=DTYPE)

        if is_update_material:
            self.qp_ddsddes = list()
            self.qp_strains = list()
            self.qp_dstrains = list()
            self.qp_stresses = list()

        for i in range(qp_number):
            if is_update_material:
                qp_weight_times_jacobi_det = qp_weight_times_jacobi_dets[i]
                qp_b_matrix_transpose = qp_b_matrices_transpose[i]
                qp_b_matrix = qp_b_matrices[i]
                qp_strain = np.dot(qp_b_matrix, element_dof_values)
                qp_dstrain = np.dot(qp_b_matrix, element_ddof_values)
                variable = {'strain': qp_strain, 'dstrain': qp_dstrain}
                qp_ddsdde, qp_output = material_data.get_tangent(variable=variable,
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
                qp_ddsdde = self.qp_ddsddes[i]
                qp_stress = self.qp_stresses[i]

            if is_update_stiffness:
                self.element_stiffness += np.dot(qp_b_matrix_transpose, np.dot(qp_ddsdde, qp_b_matrix)) * qp_weight_times_jacobi_det

            if is_update_fint:
                self.element_fint += np.dot(qp_b_matrix_transpose, qp_stress) * qp_weight_times_jacobi_det

    def update_element_field_variables(self) -> None:
        self.qp_field_variables['strain'] = np.array(self.qp_strains, dtype=DTYPE) + np.array(self.qp_dstrains, dtype=DTYPE)
        self.qp_field_variables['stress'] = np.array(self.qp_stresses, dtype=DTYPE)

        self.qp_field_variables['strain'] = np.pad(self.qp_field_variables['strain'], ((0, 0), (0, 1)), mode='constant', constant_values=0)
        self.qp_field_variables['stress'] = np.pad(self.qp_field_variables['stress'], ((0, 0), (0, 1)), mode='constant', constant_values=0)

        # for key in self.qp_state_variables_new[0].keys():
        #     if key not in ['strain', 'stress']:
        #         variable = []
        #         for qp_state_variable_new in self.qp_state_variables_new:
        #             variable.append(qp_state_variable_new[key])
        #         self.qp_field_variables[f'SDV-{key}'] = np.array(variable, dtype=DTYPE)
        self.element_nodal_field_variables = set_element_field_variables(self.qp_field_variables, self.iso_element_shape, self.dimension, self.nodes_number)


if __name__ == "__main__":
    from pyfem.utils.visualization import print_slots_dict

    print_slots_dict(CohesiveZone.__slots_dict__)

    from pyfem.Job import Job

    job = Job(r'..\..\..\examples\mechanical\cohesive\Job-1.toml')

    job.assembly.element_data_list[0].show()
