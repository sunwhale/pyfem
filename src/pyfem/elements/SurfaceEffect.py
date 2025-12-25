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


class SurfaceEffect(BaseElement):
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

    __allowed_material_data_list__ = []

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
        self.allowed_material_number = 0

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

        # print(dof.names, self.dof_names)

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

        self.create_qp_b_matrices()

    def create_qp_b_matrices(self) -> None:
        qp_number = self.iso_element_shape.qp_number
        dimension = self.dimension
        import numpy as np
        np.set_printoptions(precision=3, suppress=True, linewidth=10000)
        if self.dimension == 2:
            self.qp_b_matrices = np.zeros(shape=(self.qp_number, 2, self.element_dof_number), dtype=DTYPE)

        elif self.dimension == 3:
            v1 = self.node_coords[1, :] - self.node_coords[0, :]
            v2 = self.node_coords[2, :] - self.node_coords[0, :]

            self.normal = np.cross(v1, v2)

            normal_length = np.linalg.norm(self.normal)
            if normal_length > 1e-16:
                self.normal /= normal_length
            else:
                raise ValueError(error_style('The three points are collinear and cannot compute the normal vector.'))

            self.qp_b_matrices = np.zeros(shape=(qp_number, dimension, dimension * qp_number), dtype=DTYPE)
            for iqp, qp_shape_value in enumerate(self.iso_element_shape.qp_shape_values):
                for i, value in enumerate(qp_shape_value):
                    for j in range(self.dimension):
                        self.qp_b_matrices[iqp, j, self.dimension * i + j] = value

        self.qp_b_matrices_transpose = np.array([qp_b_matrix.transpose() for qp_b_matrix in self.qp_b_matrices])

    def update_element_fext(self, is_update_fext: bool = True) -> None:
        pressure = self.section.data_dict['pressure']
        traction = pressure * self.normal
        weight = self.qp_weight_times_jacobi_dets.reshape(-1, 1)
        if is_update_fext:
            f_ext = np.dot(self.qp_b_matrices_transpose, traction) * weight
            f_ext = np.sum(f_ext, axis=0)
            print(f_ext)


if __name__ == "__main__":
    from pyfem.utils.visualization import print_slots_dict

    print_slots_dict(SurfaceEffect.__slots_dict__)

    from pyfem.Job import Job

    job = Job(r'..\..\..\tests\1element\quad4.toml')

    # job.assembly.element_data_list[0].show()
