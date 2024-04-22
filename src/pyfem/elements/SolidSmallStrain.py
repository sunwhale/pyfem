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
    **固体小变形单元**

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

    测试：比较当前代码和商业有限元软件 ABAQUS 的刚度矩阵，可以通过修改 ABAQUS inp文件，添加以下代码，将单元刚度矩阵输出到 ELEMENTSTIFFNESS.mtx 文件中::

        *Output, history, variable=PRESELECT
        *Element Matrix Output, Elset=Part-1-1.Set-All, File Name=ElementStiffness, Output File=User Defined, stiffness=yes

    我们可以发现 ABAQUS 使用的单元刚度矩阵和当前代码计算的刚度矩阵有一定的差别，这是由于 ABAQUS 采用了 B-Bar 方法对 B 矩阵进行了修正。

    注意：当前单元均为原始形式，存在剪切自锁，体积自锁，沙漏模式和零能模式等误差模式。几种误差模式的描述可以参考 https://blog.csdn.net/YORU_NO_KUNI/article/details/130370094。
    """

    __slots_dict__: dict = {
        'qp_b_matrices': ('ndarray', '积分点处的B矩阵列表'),
        'qp_b_matrices_transpose': ('ndarray', '积分点处的B矩阵转置列表'),
        'qp_strains': ('list[ndarray]', '积分点处的应变列表'),
        'qp_stresses': ('list[ndarray]', '积分点处的应力列表'),
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

        self.allowed_material_data_list = [
            ('ElasticIsotropic', 'PlasticKinematicHardening', 'PlasticCrystal', 'PlasticCrystalGNDs', 'ViscoElasticMaxwell')]
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
        self.element_stiffness = zeros(shape=(self.element_dof_number, self.element_dof_number), dtype=DTYPE)

        self.qp_b_matrices: ndarray = None  # type: ignore
        self.qp_b_matrices_transpose: ndarray = None  # type: ignore
        self.qp_strains: list[ndarray] = None  # type: ignore
        self.qp_stresses: list[ndarray] = None  # type: ignore

        self.create_qp_b_matrices()

    def create_qp_b_matrices(self) -> None:
        if self.dimension == 2:
            self.qp_b_matrices = zeros(shape=(self.qp_number, 3, self.element_dof_number), dtype=DTYPE)
            for iqp, qp_dhdx in enumerate(self.qp_dhdxes):
                for i, val in enumerate(qp_dhdx.transpose()):
                    self.qp_b_matrices[iqp, 0, i * 2 + 0] = val[0]
                    self.qp_b_matrices[iqp, 1, i * 2 + 1] = val[1]
                    self.qp_b_matrices[iqp, 2, i * 2 + 0] = val[1]
                    self.qp_b_matrices[iqp, 2, i * 2 + 1] = val[0]

        elif self.dimension == 3:
            self.qp_b_matrices = zeros(shape=(self.iso_element_shape.qp_number, 6, self.element_dof_number), dtype=DTYPE)
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
        qp_b_matrices = self.qp_b_matrices
        qp_b_matrices_transpose = self.qp_b_matrices_transpose
        qp_weight_times_jacobi_dets = self.qp_weight_times_jacobi_dets

        qp_state_variables = self.qp_state_variables
        qp_state_variables_new = self.qp_state_variables_new

        element_dof_values = self.element_dof_values
        element_ddof_values = self.element_ddof_values

        material_data = self.material_data_list[0]

        if is_update_stiffness:
            self.element_stiffness = zeros(shape=(self.element_dof_number, self.element_dof_number), dtype=DTYPE)

        if is_update_fint:
            self.element_fint = zeros(self.element_dof_number, dtype=DTYPE)

        if is_update_material:
            self.qp_ddsddes = list()
            self.qp_strains = list()
            self.qp_stresses = list()

        for i in range(qp_number):
            if is_update_material:
                qp_weight_times_jacobi_det = qp_weight_times_jacobi_dets[i]
                qp_b_matrix_transpose = qp_b_matrices_transpose[i]
                qp_b_matrix = qp_b_matrices[i]
                qp_strain = dot(qp_b_matrix, element_dof_values)
                qp_dstrain = dot(qp_b_matrix, element_ddof_values)
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
                self.qp_stresses.append(qp_stress)
            else:
                qp_b_matrix_transpose = qp_b_matrices_transpose[i]
                qp_b_matrix = qp_b_matrices[i]
                qp_weight_times_jacobi_det = qp_weight_times_jacobi_dets[i]
                qp_ddsdde = self.qp_ddsddes[i]
                qp_stress = self.qp_stresses[i]

            if is_update_stiffness:
                self.element_stiffness += dot(qp_b_matrix_transpose, dot(qp_ddsdde, qp_b_matrix)) * \
                                          qp_weight_times_jacobi_det

            if is_update_fint:
                self.element_fint += dot(qp_b_matrix_transpose, qp_stress) * qp_weight_times_jacobi_det

    def update_element_field_variables(self) -> None:
        qp_stresses = self.qp_stresses
        qp_strains = self.qp_strains

        average_strain = average(qp_strains, axis=0)
        average_stress = average(qp_stresses, axis=0)

        self.qp_field_variables['strain'] = array(qp_strains, dtype=DTYPE)
        self.qp_field_variables['stress'] = array(qp_stresses, dtype=DTYPE)

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
