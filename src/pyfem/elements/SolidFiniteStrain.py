# -*- coding: utf-8 -*-
"""

"""
from numpy import array, zeros, dot, ndarray, average, eye, transpose
from numpy.linalg import det

from pyfem.elements.BaseElement import BaseElement
from pyfem.fem.Timer import Timer
from pyfem.fem.constants import DTYPE
from pyfem.io.Dof import Dof
from pyfem.io.Material import Material
from pyfem.io.Section import Section
from pyfem.isoelements.IsoElementShape import IsoElementShape
from pyfem.materials.get_material_data import MaterialData
from pyfem.utils.colors import error_style
from pyfem.utils.mechanics import inverse


class SolidFiniteStrain(BaseElement):
    """
    **固体有限变形单元**

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
        'qp_bnl_matrices': ('ndarray', '积分点处的非线性B矩阵列表'),
        'qp_bnl_matrices_transpose': ('ndarray', '积分点处的非线性B矩阵转置列表'),
        'qp_deformation_gradients_0': ('list[ndarray]', '积分点处的历史载荷步变形梯度列表'),
        'qp_deformation_gradients_1': ('list[ndarray]', '积分点处的当前载荷步变形梯度列表'),
        'qp_strains': ('list[ndarray]', '积分点处的应变列表'),
        'qp_dstrains': ('list[ndarray]', '积分点处的应变增量列表'),
        'qp_stresses': ('list[ndarray]', '积分点处的应力列表'),
        'qp_green_lagrange_strains_0': ('list[ndarray]', '积分点处的历史载荷步 Green-Lagrange 应变列表'),
        'qp_green_lagrange_strains_1': ('list[ndarray]', '积分点处的当前载荷步 Green-Lagrange 应变列表'),
        'qp_jacobis_t': ('ndarray(qp_number, 空间维度, 空间维度)', 'UL方法X^t构型对应的积分点处的雅克比矩阵列表'),
        'qp_jacobi_invs_t': ('ndarray(qp_number,)', 'UL方法X^t构型对应的积分点处的雅克比矩阵逆矩阵列表'),
        'qp_jacobi_dets_t': ('ndarray(qp_number,)', 'UL方法X^t构型对应的积分点处的雅克比矩阵行列式列表'),
        'qp_weight_times_jacobi_dets_t': ('ndarray(qp_number,)', 'UL方法X^t构型对应的积分点处的雅克比矩阵行列式乘以积分权重列表'),
        'method': ('string', '使用的求解格式'),
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

        self.method: str = 'TL'
        # self.method: str = 'UL'

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
        self.qp_deformation_gradients_0: ndarray = None  # type: ignore
        self.qp_deformation_gradients_1: ndarray = None  # type: ignore
        self.qp_strains: list[ndarray] = None  # type: ignore
        self.qp_dstrains: list[ndarray] = None  # type: ignore
        self.qp_stresses: list[ndarray] = None  # type: ignore
        self.qp_bnl_matrices: ndarray = None  # type: ignore
        self.qp_bnl_matrices_transpose: ndarray = None  # type: ignore
        self.qp_green_lagrange_strains_0: list[ndarray] = None  # type: ignore
        self.qp_green_lagrange_strains_1: list[ndarray] = None  # type: ignore

        # 采用 Updated Lagrangian 格式时所需要的中间变量，对应 t 时刻的 X^t
        self.qp_jacobis_t: ndarray = None  # type: ignore
        self.qp_jacobi_dets_t: ndarray = None  # type: ignore
        self.qp_jacobi_invs_t: ndarray = None  # type: ignore
        self.qp_weight_times_jacobi_dets_t: ndarray = None  # type: ignore

        self.update_kinematics()
        self.create_qp_b_matrices()
        self.create_qp_bnl_matrices()

    def cal_jacobi_t(self) -> None:
        self.qp_jacobis_t = dot(self.iso_element_shape.qp_shape_gradients,
                                self.node_coords + self.element_dof_values.reshape(-1, self.dimension)).swapaxes(1, 2)
        self.qp_jacobi_dets_t = det(self.qp_jacobis_t)
        self.qp_jacobi_invs_t = inverse(self.qp_jacobis_t, self.qp_jacobi_dets_t)
        self.qp_weight_times_jacobi_dets_t = self.iso_element_shape.qp_weights * self.qp_jacobi_dets_t

    def update_kinematics(self) -> None:
        nodes_number = self.iso_element_shape.nodes_number
        # self.element_ddof_values = array([0., 0., 0.0855, 0.176, -0.0855, 0.176, 0., 0.])
        # 计算历史变形梯度
        qp_deformation_gradients_0 = []
        dof_reshape_0 = self.element_dof_values.reshape(nodes_number, len(self.dof.names))
        for qp_shape_gradient, qp_jacobi_inv in zip(self.iso_element_shape.qp_shape_gradients, self.qp_jacobi_invs):
            qp_deformation_gradients_0.append(eye(self.dimension) + transpose(dot(dot(qp_jacobi_inv, qp_shape_gradient), dof_reshape_0)))
        self.qp_deformation_gradients_0 = array(qp_deformation_gradients_0)

        # 计算当前变形梯度
        qp_deformation_gradients_1 = []
        dof_reshape_1 = (self.element_dof_values + self.element_ddof_values).reshape(nodes_number, len(self.dof.names))
        for qp_shape_gradient, qp_jacobi_inv in zip(self.iso_element_shape.qp_shape_gradients, self.qp_jacobi_invs):
            qp_deformation_gradients_1.append(eye(self.dimension) + transpose(dot(dot(qp_jacobi_inv, qp_shape_gradient), dof_reshape_1)))
        self.qp_deformation_gradients_1 = array(qp_deformation_gradients_1)

        # 计算历史Green-Lagrange应变
        qp_green_lagrange_strains_0 = []
        for qp_deformation_gradients_0 in self.qp_deformation_gradients_0:
            qp_green_lagrange_strains_0.append(0.5 * (dot(qp_deformation_gradients_0.transpose(), qp_deformation_gradients_0) - eye(self.dimension)))
        self.qp_green_lagrange_strains_0 = array(qp_green_lagrange_strains_0)

        # 计算当前Green-Lagrange应变
        qp_green_lagrange_strains_1 = []
        for qp_deformation_gradients_1 in self.qp_deformation_gradients_1:
            qp_green_lagrange_strains_1.append(0.5 * (dot(qp_deformation_gradients_1.transpose(), qp_deformation_gradients_1) - eye(self.dimension)))
        self.qp_green_lagrange_strains_1 = array(qp_green_lagrange_strains_1)

        # 应变的向量记法
        self.qp_strains = []
        self.qp_dstrains = []
        for iqp, (qp_green_lagrange_strain_0, qp_green_lagrange_strain_1) in enumerate(zip(self.qp_green_lagrange_strains_0, self.qp_green_lagrange_strains_1)):
            if self.dimension == 2:
                qp_strain = zeros(shape=(3,))
                qp_strain[0] = qp_green_lagrange_strain_0[0, 0]
                qp_strain[1] = qp_green_lagrange_strain_0[1, 1]
                qp_strain[2] = 2.0 * qp_green_lagrange_strain_0[0, 1]
                qp_dstrain = zeros(shape=(3,))
                qp_dstrain[0] = qp_green_lagrange_strain_1[0, 0] - qp_green_lagrange_strain_0[0, 0]
                qp_dstrain[1] = qp_green_lagrange_strain_1[1, 1] - qp_green_lagrange_strain_0[1, 1]
                qp_dstrain[2] = 2.0 * qp_green_lagrange_strain_1[0, 1] - 2.0 * qp_green_lagrange_strain_0[0, 1]
            elif self.dimension == 3:
                qp_strain = zeros(shape=(6,))
                qp_strain[0] = qp_green_lagrange_strain_0[0, 0]
                qp_strain[1] = qp_green_lagrange_strain_0[1, 1]
                qp_strain[2] = qp_green_lagrange_strain_0[2, 2]
                qp_strain[3] = 2.0 * qp_green_lagrange_strain_0[0, 1]
                qp_strain[4] = 2.0 * qp_green_lagrange_strain_0[0, 2]
                qp_strain[5] = 2.0 * qp_green_lagrange_strain_0[1, 2]
                qp_dstrain = zeros(shape=(6,))
                qp_dstrain[0] = qp_green_lagrange_strain_1[0, 0] - qp_green_lagrange_strain_0[0, 0]
                qp_dstrain[1] = qp_green_lagrange_strain_1[1, 1] - qp_green_lagrange_strain_0[1, 1]
                qp_dstrain[2] = qp_green_lagrange_strain_1[2, 2] - qp_green_lagrange_strain_0[2, 2]
                qp_dstrain[3] = 2.0 * qp_green_lagrange_strain_1[0, 1] - 2.0 * qp_green_lagrange_strain_0[0, 1]
                qp_dstrain[4] = 2.0 * qp_green_lagrange_strain_1[0, 2] - 2.0 * qp_green_lagrange_strain_0[0, 2]
                qp_dstrain[5] = 2.0 * qp_green_lagrange_strain_1[1, 2] - 2.0 * qp_green_lagrange_strain_0[1, 2]
            else:
                error_msg = f'{self.dimension} is not the supported dimension'
                raise NotImplementedError(error_style(error_msg))
            self.qp_strains.append(qp_strain)
            self.qp_dstrains.append(qp_dstrain)

            # if self.element_id == 0 and iqp == 0:
            #     print(self.element_dof_values)
            #     print(self.element_ddof_values)
            #     print(self.qp_strains[iqp])
            #     print(self.qp_dstrains[iqp])

    def create_qp_b_matrices(self) -> None:
        if self.dimension == 2:
            self.qp_b_matrices = zeros(shape=(self.qp_number, 3, self.element_dof_number), dtype=DTYPE)
        elif self.dimension == 3:
            self.qp_b_matrices = zeros(shape=(self.qp_number, 6, self.element_dof_number), dtype=DTYPE)

        if self.method == "TL":
            for iqp, (qp_shape_gradient, F1, qp_jacobi_inv) in enumerate(zip(self.iso_element_shape.qp_shape_gradients,
                                                                             self.qp_deformation_gradients_1, self.qp_jacobi_invs)):
                qp_dhdx = dot(qp_shape_gradient.transpose(), qp_jacobi_inv)
                if self.dimension == 2:
                    for i, val in enumerate(qp_dhdx):
                        self.qp_b_matrices[iqp, 0, i * 2 + 0] = val[0] * F1[0, 0]
                        self.qp_b_matrices[iqp, 0, i * 2 + 1] = val[0] * F1[1, 0]
                        self.qp_b_matrices[iqp, 1, i * 2 + 0] = val[1] * F1[0, 1]
                        self.qp_b_matrices[iqp, 1, i * 2 + 1] = val[1] * F1[1, 1]
                        self.qp_b_matrices[iqp, 2, i * 2 + 0] = val[1] * F1[0, 0] + val[0] * F1[0, 1]
                        self.qp_b_matrices[iqp, 2, i * 2 + 1] = val[1] * F1[1, 0] + val[0] * F1[1, 1]
                elif self.dimension == 3:
                    for i, val in enumerate(qp_dhdx):
                        self.qp_b_matrices[iqp, 0, i * 3 + 0] = val[0] * F1[0, 0]
                        self.qp_b_matrices[iqp, 0, i * 3 + 1] = val[0] * F1[1, 0]
                        self.qp_b_matrices[iqp, 0, i * 3 + 2] = val[0] * F1[2, 0]
                        self.qp_b_matrices[iqp, 1, i * 3 + 0] = val[1] * F1[0, 1]
                        self.qp_b_matrices[iqp, 1, i * 3 + 1] = val[1] * F1[1, 1]
                        self.qp_b_matrices[iqp, 1, i * 3 + 2] = val[1] * F1[2, 1]
                        self.qp_b_matrices[iqp, 2, i * 3 + 0] = val[2] * F1[0, 2]
                        self.qp_b_matrices[iqp, 2, i * 3 + 1] = val[2] * F1[1, 2]
                        self.qp_b_matrices[iqp, 2, i * 3 + 2] = val[2] * F1[2, 2]
                        self.qp_b_matrices[iqp, 3, i * 3 + 0] = val[1] * F1[0, 0] + val[0] * F1[0, 1]
                        self.qp_b_matrices[iqp, 3, i * 3 + 1] = val[1] * F1[1, 0] + val[0] * F1[1, 1]
                        self.qp_b_matrices[iqp, 3, i * 3 + 2] = val[1] * F1[2, 0] + val[0] * F1[1, 1]
                        self.qp_b_matrices[iqp, 4, i * 3 + 0] = val[0] * F1[0, 2] + val[2] * F1[0, 0]
                        self.qp_b_matrices[iqp, 4, i * 3 + 1] = val[0] * F1[1, 2] + val[2] * F1[1, 0]
                        self.qp_b_matrices[iqp, 4, i * 3 + 2] = val[0] * F1[2, 2] + val[2] * F1[2, 0]
                        self.qp_b_matrices[iqp, 5, i * 3 + 0] = val[2] * F1[0, 1] + val[1] * F1[0, 2]
                        self.qp_b_matrices[iqp, 5, i * 3 + 1] = val[2] * F1[1, 1] + val[1] * F1[1, 2]
                        self.qp_b_matrices[iqp, 5, i * 3 + 2] = val[2] * F1[2, 1] + val[1] * F1[2, 2]

                if self.element_id == 0 and iqp == 0:
                    print(self.element_dof_values)
                    print(self.element_ddof_values)
                    print(self.qp_deformation_gradients_0[iqp])
                    print(self.qp_green_lagrange_strains_0[iqp])
                    print(self.qp_deformation_gradients_1[iqp])
                    print(self.qp_green_lagrange_strains_1[iqp])

                    # print(qp_dhdx)
                    # print(F1)
                    # print(self.qp_b_matrices[iqp])

        elif self.method == "UL":
            self.cal_jacobi_t()
            for iqp, (qp_shape_gradient, qp_jacobi_inv_t) in enumerate(zip(self.iso_element_shape.qp_shape_gradients, self.qp_jacobi_invs_t)):
                qp_dhdx_t = dot(qp_shape_gradient.transpose(), qp_jacobi_inv_t)
                if self.dimension == 2:
                    for i, val in enumerate(qp_dhdx_t):
                        self.qp_b_matrices[iqp, 0, i * 2 + 0] = val[0]
                        self.qp_b_matrices[iqp, 1, i * 2 + 1] = val[1]
                        self.qp_b_matrices[iqp, 2, i * 2 + 0] = val[1]
                        self.qp_b_matrices[iqp, 2, i * 2 + 1] = val[0]
                elif self.dimension == 3:
                    for i, val in enumerate(qp_dhdx_t):
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

    def create_qp_bnl_matrices(self) -> None:
        if self.dimension == 2:
            self.qp_bnl_matrices = zeros(shape=(self.qp_number, 4, self.element_dof_number), dtype=DTYPE)
        elif self.dimension == 3:
            self.qp_bnl_matrices = zeros(shape=(self.qp_number, 9, self.element_dof_number), dtype=DTYPE)

        if self.method == "TL":
            for iqp, (qp_shape_gradient, qp_jacobi_inv) in enumerate(zip(self.iso_element_shape.qp_shape_gradients, self.qp_jacobi_invs)):
                qp_dhdx = dot(qp_shape_gradient.transpose(), qp_jacobi_inv)
                if self.dimension == 2:
                    for i, val in enumerate(qp_dhdx):
                        self.qp_bnl_matrices[iqp, 0, i * 2 + 0] = val[0]
                        self.qp_bnl_matrices[iqp, 1, i * 2 + 0] = val[1]
                        self.qp_bnl_matrices[iqp, 2, i * 2 + 1] = val[0]
                        self.qp_bnl_matrices[iqp, 3, i * 2 + 1] = val[1]
                elif self.dimension == 3:
                    for i, val in enumerate(qp_dhdx):
                        self.qp_bnl_matrices[iqp, 0, i * 3 + 0] = val[0]
                        self.qp_bnl_matrices[iqp, 1, i * 3 + 0] = val[1]
                        self.qp_bnl_matrices[iqp, 2, i * 3 + 0] = val[2]
                        self.qp_bnl_matrices[iqp, 3, i * 3 + 1] = val[0]
                        self.qp_bnl_matrices[iqp, 4, i * 3 + 1] = val[1]
                        self.qp_bnl_matrices[iqp, 5, i * 3 + 1] = val[2]
                        self.qp_bnl_matrices[iqp, 6, i * 3 + 2] = val[0]
                        self.qp_bnl_matrices[iqp, 7, i * 3 + 2] = val[1]
                        self.qp_bnl_matrices[iqp, 8, i * 3 + 2] = val[2]

        elif self.method == "UL":
            for iqp, (qp_shape_gradient, qp_jacobi_inv_t) in enumerate(zip(self.iso_element_shape.qp_shape_gradients, self.qp_jacobi_invs_t)):
                qp_dhdx_t = dot(qp_shape_gradient.transpose(), qp_jacobi_inv_t)
                if self.dimension == 2:
                    for i, val in enumerate(qp_dhdx_t):
                        self.qp_bnl_matrices[iqp, 0, i * 2 + 0] = val[0]
                        self.qp_bnl_matrices[iqp, 1, i * 2 + 0] = val[1]
                        self.qp_bnl_matrices[iqp, 2, i * 2 + 1] = val[0]
                        self.qp_bnl_matrices[iqp, 3, i * 2 + 1] = val[1]
                elif self.dimension == 3:
                    for i, val in enumerate(qp_dhdx_t):
                        self.qp_bnl_matrices[iqp, 0, i * 3 + 0] = val[0]
                        self.qp_bnl_matrices[iqp, 1, i * 3 + 0] = val[1]
                        self.qp_bnl_matrices[iqp, 2, i * 3 + 0] = val[2]
                        self.qp_bnl_matrices[iqp, 3, i * 3 + 1] = val[0]
                        self.qp_bnl_matrices[iqp, 4, i * 3 + 1] = val[1]
                        self.qp_bnl_matrices[iqp, 5, i * 3 + 1] = val[2]
                        self.qp_bnl_matrices[iqp, 6, i * 3 + 2] = val[0]
                        self.qp_bnl_matrices[iqp, 7, i * 3 + 2] = val[1]
                        self.qp_bnl_matrices[iqp, 8, i * 3 + 2] = val[2]

        self.qp_bnl_matrices_transpose = array([qp_bnl_matrix.transpose() for qp_bnl_matrix in self.qp_bnl_matrices])

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
        qp_bnl_matrices = self.qp_bnl_matrices
        qp_bnl_matrices_transpose = self.qp_bnl_matrices_transpose
        qp_weight_times_jacobi_dets = self.qp_weight_times_jacobi_dets

        qp_state_variables = self.qp_state_variables
        qp_state_variables_new = self.qp_state_variables_new

        material_data = self.material_data_list[0]

        if is_update_stiffness:
            self.element_stiffness = zeros(shape=(self.element_dof_number, self.element_dof_number), dtype=DTYPE)

        if is_update_fint:
            self.element_fint = zeros(self.element_dof_number, dtype=DTYPE)

        if is_update_material:
            self.qp_ddsddes = list()
            self.qp_strains = list()
            self.qp_stresses = list()
            self.update_kinematics()
            self.create_qp_b_matrices()

        for i in range(qp_number):
            qp_weight_times_jacobi_det = qp_weight_times_jacobi_dets[i]
            qp_b_matrix_transpose = qp_b_matrices_transpose[i]
            qp_bnl_matrix_transpose = qp_bnl_matrices_transpose[i]
            qp_b_matrix = qp_b_matrices[i]
            qp_bnl_matrix = qp_bnl_matrices[i]
            if is_update_material:
                qp_strain = self.qp_strains[i]
                qp_dstrain = self.qp_dstrains[i]
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
                # if self.element_id == 0 and i == 0:
                #     print(qp_strain)
                #     print(self.element_ddof_values)
            else:
                qp_ddsdde = self.qp_ddsddes[i]
                qp_stress = self.qp_stresses[i]

            if is_update_stiffness:
                qp_stress_matrix = self.vogit_to_block_diagonal_matrix(qp_stress)
                if self.method == 'TL':
                    self.element_stiffness += dot(qp_b_matrix_transpose, dot(qp_ddsdde, qp_b_matrix)) * qp_weight_times_jacobi_det
                    self.element_stiffness += dot(qp_bnl_matrix_transpose, dot(qp_stress_matrix, qp_bnl_matrix)) * qp_weight_times_jacobi_det
                elif self.method == 'UL':
                    self.element_stiffness += dot(qp_b_matrix_transpose, dot(qp_ddsdde, qp_b_matrix)) * self.qp_weight_times_jacobi_dets_t[i]
                    self.element_stiffness += dot(qp_bnl_matrix_transpose, dot(qp_stress_matrix, qp_bnl_matrix)) * self.qp_weight_times_jacobi_dets_t[i]

            if is_update_fint:
                if self.method == 'TL':
                    self.element_fint += dot(qp_b_matrix_transpose, qp_stress) * qp_weight_times_jacobi_det
                elif self.method == 'UL':
                    self.element_fint += dot(qp_b_matrix_transpose, qp_stress) * self.qp_weight_times_jacobi_dets_t[i]

        # if self.element_id == 0:
        #     print(self.element_fint)

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

    def vogit_to_block_diagonal_matrix(self, stress):
        T = zeros(shape=(self.dimension * self.dimension, self.dimension * self.dimension))

        if self.dimension == 2:
            T[0, 0] = stress[0]
            T[1, 1] = stress[1]
            T[0, 1] = stress[2]
            T[1, 0] = stress[2]
            T[self.dimension:, self.dimension:] = T[:self.dimension, :self.dimension]

        elif self.dimension == 3:
            T[0, 0] = stress[0]
            T[1, 1] = stress[1]
            T[2, 2] = stress[2]
            T[0, 1] = stress[3]
            T[0, 2] = stress[4]
            T[1, 2] = stress[5]
            T[1, 0] = stress[3]
            T[2, 0] = stress[4]
            T[2, 1] = stress[5]
            T[self.dimension:2 * self.dimension, self.dimension:2 * self.dimension] = T[:self.dimension, :self.dimension]
            T[2 * self.dimension:, 2 * self.dimension:] = T[:self.dimension, :self.dimension]

        return T


if __name__ == "__main__":
    # from pyfem.utils.visualization import print_slots_dict
    #
    # print_slots_dict(SolidFiniteStrain.__slots_dict__)

    from pyfem.Job import Job

    job = Job(r'..\..\..\examples\mechanical\1element\quad4\Job-1.toml')

    # job.assembly.element_data_list[0].show()

    e = SolidFiniteStrain(element_id=job.assembly.element_data_list[0].element_id,
                          iso_element_shape=job.assembly.element_data_list[0].iso_element_shape,
                          connectivity=job.assembly.element_data_list[0].connectivity,
                          node_coords=job.assembly.element_data_list[0].node_coords,
                          dof=job.assembly.element_data_list[0].dof,
                          materials=job.assembly.element_data_list[0].materials,
                          section=job.assembly.element_data_list[0].section,
                          material_data_list=job.assembly.element_data_list[0].material_data_list,
                          timer=job.assembly.element_data_list[0].timer)

    e.show()
