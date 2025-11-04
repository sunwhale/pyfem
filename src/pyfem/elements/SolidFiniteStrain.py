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
from pyfem.utils.mechanics import inverse


class SolidFiniteStrain(BaseElement):
    """
    **固体有限变形单元**

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

    测试：比较当前代码和商业有限元软件 ABAQUS 的刚度矩阵，可以通过修改 ABAQUS inp文件，添加以下代码，将单元刚度矩阵输出到 ELEMENTSTIFFNESS.mtx 文件中::

        *Output, history, variable=PRESELECT
        *Element Matrix Output, Elset=Part-1-1.Set-All, File Name=ElementStiffness, Output File=User Defined, stiffness=yes

    我们可以发现 ABAQUS 使用的单元刚度矩阵和当前代码计算的刚度矩阵有一定的差别，这是由于 ABAQUS 采用了 B-Bar 方法对 B 矩阵进行了修正。

    注意：当前单元均为原始形式，存在剪切自锁，体积自锁，沙漏模式和零能模式等误差模式。几种误差模式的描述可以参考 https://blog.csdn.net/YORU_NO_KUNI/article/details/130370094。
    """

    __slots_dict__: dict = {
        'qp_b_matrices': ('np.ndarray', '积分点处的B矩阵列表'),
        'qp_b_matrices_transpose': ('np.ndarray', '积分点处的B矩阵转置列表'),
        'qp_bnl_matrices': ('np.ndarray', '积分点处的非线性B矩阵列表'),
        'qp_bnl_matrices_transpose': ('np.ndarray', '积分点处的非线性B矩阵转置列表'),
        'qp_deformation_gradients_0': ('list[np.ndarray]', '积分点处的历史载荷步变形梯度列表'),
        'qp_deformation_gradients_1': ('list[np.ndarray]', '积分点处的当前载荷步变形梯度列表'),
        'qp_strains': ('list[np.ndarray]', '积分点处的应变列表'),
        'qp_dstrains': ('list[np.ndarray]', '积分点处的应变增量列表'),
        'qp_stresses': ('list[np.ndarray]', '积分点处的应力列表'),
        'qp_green_lagrange_strains_0': ('list[np.ndarray]', '积分点处的历史载荷步 Green-Lagrange 应变列表'),
        'qp_green_lagrange_strains_1': ('list[np.ndarray]', '积分点处的当前载荷步 Green-Lagrange 应变列表'),
        'qp_jacobis_t': ('np.ndarray(qp_number, 空间维度, 空间维度)', 'UL方法X^t构型对应的积分点处的雅克比矩阵列表'),
        'qp_jacobi_invs_t': ('np.ndarray(qp_number,)', 'UL方法X^t构型对应的积分点处的雅克比矩阵逆矩阵列表'),
        'qp_jacobi_dets_t': ('np.ndarray(qp_number,)', 'UL方法X^t构型对应的积分点处的雅克比矩阵行列式列表'),
        'qp_weight_times_jacobi_dets_t': ('np.ndarray(qp_number,)', 'UL方法X^t构型对应的积分点处的雅克比矩阵行列式乘以积分权重列表'),
        'method': ('string', '使用的求解格式'),
        'ntens': ('int', '总应力数量'),
        'ndi': ('int', '轴向应力数量'),
        'nshr': ('int', '剪切应力数量')
    }

    __slots__: list = BaseElement.__slots__ + [slot for slot in __slots_dict__.keys()]

    __allowed_material_data_list__ = [('ElasticIsotropic', 'PlasticKinematicHardening', 'PlasticCrystal', 'PlasticCrystalGNDs', 'ViscoElasticMaxwell', 'User')]

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
        self.element_dof_values = np.zeros(element_dof_number, dtype=DTYPE)
        self.element_ddof_values = np.zeros(element_dof_number, dtype=DTYPE)
        self.element_fint = np.zeros(element_dof_number, dtype=DTYPE)
        self.element_stiffness = np.zeros(shape=(self.element_dof_number, self.element_dof_number), dtype=DTYPE)

        self.qp_b_matrices: np.ndarray = None  # type: ignore
        self.qp_b_matrices_transpose: np.ndarray = None  # type: ignore
        self.qp_deformation_gradients_0: np.ndarray = None  # type: ignore
        self.qp_deformation_gradients_1: np.ndarray = None  # type: ignore
        self.qp_strains: list[np.ndarray] = None  # type: ignore
        self.qp_dstrains: list[np.ndarray] = None  # type: ignore
        self.qp_stresses: list[np.ndarray] = None  # type: ignore
        self.qp_bnl_matrices: np.ndarray = None  # type: ignore
        self.qp_bnl_matrices_transpose: np.ndarray = None  # type: ignore
        self.qp_green_lagrange_strains_0: list[np.ndarray] = None  # type: ignore
        self.qp_green_lagrange_strains_1: list[np.ndarray] = None  # type: ignore

        # 采用 Updated Lagrangian 格式时所需要的中间变量，对应 t 时刻的 X^t
        self.qp_jacobis_t: np.ndarray = None  # type: ignore
        self.qp_jacobi_dets_t: np.ndarray = None  # type: ignore
        self.qp_jacobi_invs_t: np.ndarray = None  # type: ignore
        self.qp_weight_times_jacobi_dets_t: np.ndarray = None  # type: ignore

        self.update_kinematics()
        self.create_qp_b_matrices()
        self.create_qp_bnl_matrices()

    def cal_jacobi_t(self) -> None:
        r"""
        **计算t时刻雅克比矩阵**

        采用 Updated Lagrangian 方法需要计算 :math:`X^t` 构型下，单元积分点处的雅克比矩阵 :py:attr:`qp_jacobis_t`，雅克比矩阵的逆矩阵 :py:attr:`qp_jacobi_invs_t`，
        雅克比矩阵行列式 :py:attr:`qp_jacobi_dets_t` 和雅克比矩阵行列式乘以积分点权重 :py:attr:`qp_weight_times_jacobi_dets_t`。

        全局坐标系 :math:`\left( {{x_1},{x_2},{x_3}} \right)` 和局部坐标系 :math:`\left( {{\xi _1},{\xi _2},{\xi _3}} \right)` 之间的雅克比矩阵如下：

        .. math::
            \left\{ {\begin{array}{*{20}{c}}
              {{\text{d}}{x_1}} \\
              {{\text{d}}{x_2}} \\
              {{\text{d}}{x_3}}
            \end{array}} \right\} = \left[ {\begin{array}{*{20}{c}}
              {\frac{{\partial {x_1}}}{{\partial {\xi _1}}}}&{\frac{{\partial {x_1}}}{{\partial {\xi _2}}}}&{\frac{{\partial {x_1}}}{{\partial {\xi _3}}}} \\
              {\frac{{\partial {x_2}}}{{\partial {\xi _1}}}}&{\frac{{\partial {x_2}}}{{\partial {\xi _2}}}}&{\frac{{\partial {x_2}}}{{\partial {\xi _3}}}} \\
              {\frac{{\partial {x_3}}}{{\partial {\xi _1}}}}&{\frac{{\partial {x_3}}}{{\partial {\xi _2}}}}&{\frac{{\partial {x_3}}}{{\partial {\xi _3}}}}
            \end{array}} \right]\left\{ {\begin{array}{*{20}{c}}
              {{\text{d}}{\xi _1}} \\
              {{\text{d}}{\xi _2}} \\
              {{\text{d}}{\xi _3}}
            \end{array}} \right\}

        .. math::
            \left[ J \right] = \left[ {\begin{array}{*{20}{c}}
              {\frac{{\partial {x_1}}}{{\partial {\xi _1}}}}&{\frac{{\partial {x_1}}}{{\partial {\xi _2}}}}&{\frac{{\partial {x_1}}}{{\partial {\xi _3}}}} \\
              {\frac{{\partial {x_2}}}{{\partial {\xi _1}}}}&{\frac{{\partial {x_2}}}{{\partial {\xi _2}}}}&{\frac{{\partial {x_2}}}{{\partial {\xi _3}}}} \\
              {\frac{{\partial {x_3}}}{{\partial {\xi _1}}}}&{\frac{{\partial {x_3}}}{{\partial {\xi _2}}}}&{\frac{{\partial {x_3}}}{{\partial {\xi _3}}}}
            \end{array}} \right]

        笛卡尔全局坐标系 :math:`\left( x,y,z \right)` 和局部坐标系 :math:`\left( {\xi ,\eta ,\zeta } \right)` 之间雅克比矩阵可以表示为：

        .. math::
            \left\{ {\begin{array}{*{20}{c}}
              {{\text{d}}x} \\
              {{\text{d}}y} \\
              {{\text{d}}z}
            \end{array}} \right\} = \left[ {\begin{array}{*{20}{c}}
              {\frac{{\partial x}}{{\partial \xi }}}&{\frac{{\partial x}}{{\partial \eta }}}&{\frac{{\partial x}}{{\partial \zeta }}} \\
              {\frac{{\partial y}}{{\partial \xi }}}&{\frac{{\partial y}}{{\partial \eta }}}&{\frac{{\partial y}}{{\partial \zeta }}} \\
              {\frac{{\partial z}}{{\partial \xi }}}&{\frac{{\partial z}}{{\partial \eta }}}&{\frac{{\partial z}}{{\partial \zeta }}}
            \end{array}} \right]\left\{ {\begin{array}{*{20}{c}}
              {{\text{d}}\xi } \\
              {{\text{d}}\eta } \\
              {{\text{d}}\zeta }
            \end{array}} \right\}

        .. math::
            \left[ J \right] = \left[ {\begin{array}{*{20}{c}}
              {\frac{{\partial x}}{{\partial \xi }}}&{\frac{{\partial x}}{{\partial \eta }}}&{\frac{{\partial x}}{{\partial \zeta }}} \\
              {\frac{{\partial y}}{{\partial \xi }}}&{\frac{{\partial y}}{{\partial \eta }}}&{\frac{{\partial y}}{{\partial \zeta }}} \\
              {\frac{{\partial z}}{{\partial \xi }}}&{\frac{{\partial z}}{{\partial \eta }}}&{\frac{{\partial z}}{{\partial \zeta }}}
            \end{array}} \right]

        在 :math:`X^t` 时刻构型单元节点坐标表示：

        .. math::
            \left\{ {\begin{array}{*{20}{c}}
              {{\text{d}}{x^t}} \\
              {{\text{d}}{y^t}} \\
              {{\text{d}}{z^t}}
            \end{array}} \right\} = \left[ {\begin{array}{*{20}{c}}
              {\frac{{\partial {x^t}}}{{\partial \xi }}}&{\frac{{\partial {x^t}}}{{\partial \eta }}}&{\frac{{\partial {x^t}}}{{\partial \zeta }}} \\
              {\frac{{\partial {y^t}}}{{\partial \xi }}}&{\frac{{\partial {y^t}}}{{\partial \eta }}}&{\frac{{\partial {y^t}}}{{\partial \zeta }}} \\
              {\frac{{\partial {z^t}}}{{\partial \xi }}}&{\frac{{\partial {z^t}}}{{\partial \eta }}}&{\frac{{\partial {z^t}}}{{\partial \zeta }}}
            \end{array}} \right]\left\{ {\begin{array}{*{20}{c}}
              {{\text{d}}\xi } \\
              {{\text{d}}\eta } \\
              {{\text{d}}\zeta }
            \end{array}} \right\}

        根据单元形函数的性质有，

        .. math::
            {x^t} = \sum\limits_{k = 1}^n {{N_k}} x_k^t \\
            {y^t} = \sum\limits_{k = 1}^n {{N_k}} y_k^t \\
            {z^t} = \sum\limits_{k = 1}^n {{N_k}} z_k^t \\

        其中 :math:`n` 为单元节点总数，可以得到：

        .. math::
            \left[ J \right] = \left[ {\begin{array}{*{20}{c}}
              {\sum\limits_{i = 1}^n {\frac{{\partial {N_i}}}{{\partial \xi }}} x_i^t}&{\sum\limits_{i = 1}^n {\frac{{\partial {N_i}}}{{\partial \eta }}} x_i^t}&{\sum\limits_{i = 1}^n {\frac{{\partial {N_i}}}{{\partial \zeta }}} x_i^t} \\
              {\sum\limits_{i = 1}^n {\frac{{\partial {N_i}}}{{\partial \xi }}} y_i^t}&{\sum\limits_{i = 1}^n {\frac{{\partial {N_i}}}{{\partial \eta }}} y_i^t}&{\sum\limits_{i = 1}^n {\frac{{\partial {N_i}}}{{\partial \zeta }}} y_i^t} \\
              {\sum\limits_{i = 1}^n {\frac{{\partial {N_i}}}{{\partial \xi }}} z_i^t}&{\sum\limits_{i = 1}^n {\frac{{\partial {N_i}}}{{\partial \eta }}} z_i^t}&{\sum\limits_{i = 1}^n {\frac{{\partial {N_i}}}{{\partial \zeta }}} z_i^t}
            \end{array}} \right] = {\left( {\underbrace {\left[ {\begin{array}{*{20}{c}}
              {\frac{{\partial {N_1}}}{{\partial \xi }}}& \cdots &{\frac{{\partial {N_n}}}{{\partial \xi }}} \\
              {\frac{{\partial {N_1}}}{{\partial \eta }}}& \cdots &{\frac{{\partial {N_n}}}{{\partial \eta }}} \\
              {\frac{{\partial {N_1}}}{{\partial \zeta }}}& \cdots &{\frac{{\partial {N_n}}}{{\partial \zeta }}}
            \end{array}} \right] \cdot }_{{\text{qp_shape_gradient}}}\underbrace {\left[ {\begin{array}{*{20}{c}}
              {x_1^t}&{y_1^t}&{z_1^t} \\
               \vdots & \vdots & \vdots  \\
              {x_n^t}&{x_n^t}&{x_n^t}
            \end{array}} \right]}_{{\text{node_coords}}}} \right)^T}

        """
        self.qp_jacobis_t = np.dot(self.iso_element_shape.qp_shape_gradients,
                                   self.node_coords + self.element_dof_values.reshape(-1, self.dimension)).swapaxes(1, 2)
        self.qp_jacobi_dets_t = np.linalg.det(self.qp_jacobis_t)
        self.qp_jacobi_invs_t = inverse(self.qp_jacobis_t, self.qp_jacobi_dets_t)
        self.qp_weight_times_jacobi_dets_t = self.iso_element_shape.qp_weights * self.qp_jacobi_dets_t

    def update_kinematics(self) -> None:
        r"""
        **更新动力学参数**

        载荷步的初始时刻（ :math:`X^t` ）构形对应的单元所有积分点处的历史变形梯度矩阵 :py:attr:`qp_deformation_gradients_0` ，历史Green-Lagrange应变矩阵 :py:attr:`qp_green_lagrange_strains_0` ，

        当前增量时刻（ :math:`X^{t + \Delta t}` ）构形对应的单元所有积分点处的当前变形梯度矩阵 :py:attr:`qp_deformation_gradients_1` ，当前Green-Lagrange应变矩阵 :py:attr:`qp_green_lagrange_strains_1` 。

        单元内任意一点在 :math:`X^0` 构形和 :math:`X^t` 构形对应的坐标、位移和位移增量可通过形函数、节点坐标和节点位移表示为：

        .. math::
            \begin{gathered}
              ^0{x_i} = \sum\limits_{k = 1}^n {{N_k}} {\;^0}x_i^k,{\text{ }}i = 1,2,3 \hfill \\
              ^t{x_i} = \sum\limits_{k = 1}^n {{N_k}} {\;^t}x_i^k,{\text{ }}i = 1,2,3 \hfill \\
              ^t{u_i} = \sum\limits_{k = 1}^n {{N_k}} {\;^t}u_i^k,{\text{ }}i = 1,2,3 \hfill \\
              \Delta {u_i} = \sum\limits_{k = 1}^n {{N_k}} \;\Delta u_i^k,{\text{ }}i = 1,2,3 \hfill \\
            \end{gathered}

        其中， :math:`n` 为单元节点总数。

        ----------------------------------------
        1. 变形梯度张量计算
        ----------------------------------------

        变形梯度的矩阵形式记为：

        .. math::
            \left[ F \right] = \left[ {\begin{array}{*{20}{c}}
              {{F_{11}}}&{{F_{12}}}&{{F_{13}}} \\
              {{F_{21}}}&{{F_{22}}}&{{F_{23}}} \\
              {{F_{31}}}&{{F_{32}}}&{{F_{33}}}
            \end{array}} \right]

        已知，使用位移向量 :math:`\mathbf{u}` 表示的 :math:`X^t` 时刻构形的变形梯度为：

        .. math::
            {\mathbf{F}} = \frac{{\partial {\;^t}{\mathbf{x}}}}{{\partial {\mathbf{X}}}} =
            {\mathbf{I}} + \frac{{\partial {\;^t}{\mathbf{u}}}}{{\partial {\mathbf{X}}}}

        其中， :math:`\mathbf{X}` 是变形构型中材料点空间坐标的向量。 :math:`{}^t \mathbf{x}=\mathbf{\mathbf{X}}+{}^t \mathbf{u}` 。以分量形式表示记为：

        .. math::
            {F_{ij}} = \frac{{\partial \left( {{X_i}{ + ^t}{u_i}} \right)}}{{\partial {X_j}}} = {\delta _{ij}} +
            \frac{{\partial {\;^t}{u_i}}}{{\partial {X_j}}} = {\delta _{ij}} + {\;^t}{l_{ij}}

        其中， :math:`^t{l_{ij}}` 是位移梯度张量，

        .. math::
            ^t{l_{ij}} = \frac{{\partial {\;^t}{u_i}}}{{\partial {X_j}}} = \sum\limits_{k = 1}^n {{}_0{N_{k,j}}{\;}^tu_i^k}

        整理后发现

        .. math::
            \left( {{\delta _{11}} + {l_{11}}} \right) = {F_{11}},\left( {{\delta _{21}} + {l_{21}}} \right) =
            0 + {l_{21}} = {F_{21}}{\text{,}}\left( {{\delta _{31}} + {l_{31}}} \right) = 0 + {l_{31}} = {F_{31}} \cdots

        使用位移向量 :math:`\mathbf{u}` 表示当前时刻即， :math:`X^{{t{\text{ + }}\Delta t}}` 时刻构形变形梯度为：

        .. math::
            {\mathbf{F}} = \frac{{\partial {\;^{t{\text{ + }}\Delta t}}{\mathbf{x}}}}{{\partial {\mathbf{X}}}} =
            {\mathbf{I}} + \frac{{\partial {\;^{t{\text{ + }}\Delta t}}{\mathbf{u}}}}{{\partial {\mathbf{X}}}}

        以分量形式表示记为：

        .. math::
            {F_{ij}} = \frac{{\partial \left( {{X_i}{ + ^{t{\text{ + }}\Delta t}}{u_i}} \right)}}{{\partial {X_j}}} =
            {\delta _{ij}} + \frac{{\partial {\;^{t{\text{ + }}\Delta t}}{u_i}}}{{\partial {X_j}}} = {\delta _{ij}}{ + ^{t{\text{ + }}\Delta t}}{l_{ij}}

        其中， :math:`X^{t + \Delta t}` 时刻的位移梯度张量 :math:`{}^{t + \Delta t}{l_{ij}}` 表示为：

        .. math::
            ^{t + \Delta t}{l_{ij}} = \frac{{\partial {\;^{t + \Delta t}}{u_i}}}{{\partial {X_j}}} = \sum\limits_{k = 1}^n {{}_0{N_{k,j}}{\;}^{t + \Delta t}u_i^k}

        ----------------------------------------
        2. Green–Lagrange应变张量计算
        ----------------------------------------

        使用变形梯度张量 :math:`\mathbf{F}` 表示的 Green–Lagrange 应变张量 :math:`\mathbf{E}` 为：

        .. math::
            {\mathbf{E}} = \frac{{{{\mathbf{F}}^{\text{T}}} \cdot {\mathbf{F}} - {\mathbf{I}}}}{2}

        此处，只需使用对应的历史变形梯度矩阵qp_deformation_gradients_0和当前变形梯度矩阵qp_deformation_gradients_t即可计算得到对应的
        历史Green-Lagrange应变矩阵 qp_green_lagrange_strains_0和当前Green-Lagrange应变矩阵 qp_green_lagrange_strains_t。

        ----------------------------------------------------------------------
        3. 工程 Green–Lagrange 应变张量和应变增量的 Vogit 记法
        ----------------------------------------------------------------------

        **历史时刻**，即 :math:`X^t` 时刻构形对应的单元工程 Green-Lagrange 应变矩阵的Vogit向量表示记为 qp_strains 矩阵：

        （1）二维

        .. math::
            _0{\{ E\} ^T} = \left[ {\begin{array}{*{20}{c}}
              {{}_0{E_{11}}}&{{}_0{E_{22}}}&{{2{}_0}{E_{12}}}
            \end{array}} \right]

        对应的第二基尔霍夫(Kirchhoff)应力向量表示记为：

        .. math::
            {{}_0^t} {\{ \bar S\} ^T} = \left[ {\begin{array}{*{20}{c}}
              {{}_0^t{S_{11}}}&{{}_0^t{S_{22}}}&{{}_0^t{S_{12}}}
            \end{array}} \right]

        （2）三维

        .. math::
            _0{\{ E\} ^T} = \left[ {\begin{array}{*{20}{l}}
              {{}_0{E_{11}}}&{{}_0{E_{22}}}&{{}_0{E_{33}}}&{{2{}_0}{E_{12}}}&{{2{}_0}{E_{13}}}&{{2{}_0}{E_{23}}}
            \end{array}} \right]

        对应的第二基尔霍夫(Kirchhoff)应力向量表示记为：

        .. math::
            {}_0^t{\{ \bar S\} ^T} = \left[ {\begin{array}{*{20}{l}}
              {{}_0^t{S_{11}}}&{{}_0^t{S_{22}}}&{{}_0^t{S_{33}}}&{{}_0^t{S_{12}}}&{{}_0^t{S_{13}}}&{{}_0^t{S_{23}}}
            \end{array}} \right]

        然后使用历史Green-Lagrange应变矩阵 qp_green_lagrange_strains_0和当前Green-Lagrange应变矩阵 qp_green_lagrange_strains_t即可
        得到 :math:`X^t` 时刻构形对应的单元工程 Green–Lagrange 应变增量 qp_dstrains 的Vogit记法。

        .. math::
            {\Delta {}_0}\{ E\} { = {}^{t+\Delta t}}\left( {{}_0\{ E\} } \right) - \; {}^t \left( {{}_0\{ E\} } \right)

        若采用有限变形情况下的 U.L. 公式，则需要将参考构形下定义的第二基尔霍夫(Kirchhoff)应力张量和Green-Lagrange应变张量转换为
        当前构形下定义的 Cauchy 应力张量和 Almansi 应变张量， :math:`X^t` 时刻构形对应的单元应变矩阵的Vogit向量表示记为：

        （1）二维

        .. math::
            {}_t{\{ e\} ^T} = \left[ {\begin{array}{*{20}{c}}
              {{}_t{e_{11}}}&{{}_t{e_{22}}}&{{2{}_t}{e_{12}}}
            \end{array}} \right]

        对应的Cauchy 应力向量表示记为：

        .. math::
            {}^t{\{ \bar \sigma \} ^T} = \left[ {\begin{array}{*{20}{c}}
              {{}^t{\sigma_{11}}}&{{}^t{\sigma_{22}}}&{{}^t{\sigma_{12}}}
            \end{array}} \right]

        （2）三维

        .. math::
            {}_t{\{ e\} ^T} = \left[ {\begin{array}{*{20}{l}}
              {{}_t{e_{11}}}&{{}_t{e_{22}}}&{{}_t{e_{33}}}&{{2{}_t}{e_{12}}}&{{2{}_t}{e_{13}}}&{{2{}_t}{e_{23}}}
            \end{array}} \right]

        对应的 Cauchy 应力向量表示记为：

        .. math::
            {}^t{\{ \bar \sigma \} ^T} = {[{}^t}{\sigma_{11}}{\;^t}{\sigma_{22}}{\;^t}{\sigma_{33}}{\;^t}{\sigma_{12}}{\;^t}{\sigma_{23}}{\;^t}{\sigma_{13}}]

         :math:`X^t` 时刻构形对应的单元应变增量 qp_dstrains 的 Vogit 记法：

        .. math::
            {\Delta {}_t}\{ e\} { = {}^{t+\Delta t}}\left( {{}_t\{ e\} } \right) - \; {}^t \left( {{}_t\{ e\} } \right)

        """
        nodes_number = self.iso_element_shape.nodes_number
        # 计算历史变形梯度
        qp_deformation_gradients_0 = []
        dof_reshape_0 = self.element_dof_values.reshape(nodes_number, len(self.dof.names))
        for qp_shape_gradient, qp_jacobi_inv in zip(self.iso_element_shape.qp_shape_gradients, self.qp_jacobi_invs):
            qp_deformation_gradients_0.append(np.eye(self.dimension) + np.transpose(np.dot(np.dot(qp_jacobi_inv, qp_shape_gradient), dof_reshape_0)))
        self.qp_deformation_gradients_0 = np.array(qp_deformation_gradients_0)

        # 计算当前变形梯度
        qp_deformation_gradients_1 = []
        dof_reshape_1 = (self.element_dof_values + self.element_ddof_values).reshape(nodes_number, len(self.dof.names))
        for qp_shape_gradient, qp_jacobi_inv in zip(self.iso_element_shape.qp_shape_gradients, self.qp_jacobi_invs):
            qp_deformation_gradients_1.append(np.eye(self.dimension) + np.transpose(np.dot(np.dot(qp_jacobi_inv, qp_shape_gradient), dof_reshape_1)))
        self.qp_deformation_gradients_1 = np.array(qp_deformation_gradients_1)

        # 计算历史Green-Lagrange应变
        self.qp_green_lagrange_strains_0 = []
        for qp_deformation_gradient_0 in self.qp_deformation_gradients_0:
            self.qp_green_lagrange_strains_0.append(0.5 * (np.dot(qp_deformation_gradient_0.transpose(), qp_deformation_gradient_0) - np.eye(self.dimension)))
        # self.qp_green_lagrange_strains_0 = array(qp_green_lagrange_strains_0)

        # 计算当前Green-Lagrange应变
        self.qp_green_lagrange_strains_1 = []
        for qp_deformation_gradient_1 in self.qp_deformation_gradients_1:
            self.qp_green_lagrange_strains_1.append(0.5 * (np.dot(qp_deformation_gradient_1.transpose(), qp_deformation_gradient_1) - np.eye(self.dimension)))
        # self.qp_green_lagrange_strains_1 = array(qp_green_lagrange_strains_1)

        # X^t 时刻构形对应的单元应变的向量记法
        self.qp_strains = []
        self.qp_dstrains = []
        for iqp, (qp_green_lagrange_strain_0, qp_green_lagrange_strain_1) in enumerate(zip(self.qp_green_lagrange_strains_0, self.qp_green_lagrange_strains_1)):
            if self.dimension == 2:
                qp_strain = np.zeros(shape=(3,))
                qp_strain[0] = qp_green_lagrange_strain_0[0, 0]
                qp_strain[1] = qp_green_lagrange_strain_0[1, 1]
                qp_strain[2] = 2.0 * qp_green_lagrange_strain_0[0, 1]
                qp_dstrain = np.zeros(shape=(3,))
                qp_dstrain[0] = qp_green_lagrange_strain_1[0, 0] - qp_green_lagrange_strain_0[0, 0]
                qp_dstrain[1] = qp_green_lagrange_strain_1[1, 1] - qp_green_lagrange_strain_0[1, 1]
                qp_dstrain[2] = 2.0 * qp_green_lagrange_strain_1[0, 1] - 2.0 * qp_green_lagrange_strain_0[0, 1]
            elif self.dimension == 3:
                qp_strain = np.zeros(shape=(6,))
                qp_strain[0] = qp_green_lagrange_strain_0[0, 0]
                qp_strain[1] = qp_green_lagrange_strain_0[1, 1]
                qp_strain[2] = qp_green_lagrange_strain_0[2, 2]
                qp_strain[3] = 2.0 * qp_green_lagrange_strain_0[0, 1]
                qp_strain[4] = 2.0 * qp_green_lagrange_strain_0[0, 2]
                qp_strain[5] = 2.0 * qp_green_lagrange_strain_0[1, 2]
                qp_dstrain = np.zeros(shape=(6,))
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

    def create_qp_b_matrices(self) -> None:
        r"""
        **获得 Lagrangian 网格的线性应变-位移矩阵**

        ========================================
        Lagrangian网格
        ========================================

        ----------------------------------------
        1. 引言
        ----------------------------------------

        在 Lagrangian网格中，节点和单元随着材料移动。边界和接触面与单元的边缘保持一致，因此它们的处理较为简单。积分点也随着材料移动，因此本构方程总是在相同材料点处赋值，这对于历史相关材料是有利的。
        基于这些原因，在固体力学中广泛地应用Lagrangian网格。应用Lagrangian网格的有限元离散通常划分为更新的Lagrangian格式（U.L.）和完全的Lagrangian格式（T.L.）。这两种格式都采用了Lagrangian描述，即相关变量是材料(Lagrangian)坐标和时间的函数。
        在更新的Lagrangian格式中，导数是相对于空间(Eulerian)坐标的，弱形式包括在整个变形（或当前)构形上的积分。
        在完全的Lagrangian格式中，弱形式包括在初始(参考)构形上的积分，导数是相对于材料坐标的[1]。

        单元使用的节点坐标和位移插值是：

        .. math::
            {}^0{x_i} = \sum\limits_{k = 1}^n {{N_k}} {\;^0}x_i^k,\; {{}^t}{x_i} = \sum\limits_{k = 1}^n {{N_k}} {\;^t}x_i^k \;\;(i = 1,2,3)

        .. math::
            {}^t{u_i} = \sum\limits_{k = 1}^n {{N_k}} {\;^t}u_i^k,\; \Delta {u_i} = \sum\limits_{k = 1}^n {{N_k}} \;\Delta u_i^k \;\;(i = 1,2,3)

        这里， :math:`N_{k}` 是单元插值函数， :math:`k` 是节点个数。现推导 T. L. 和 U. L. 公式中有关单元的矩阵。

        说明：方程变量分量形式的左下标代表构形，左上标代表时刻，右上标代表哑标，右下标是变量分量指标。
        如 :math:`{}_t^{t + \Delta t}\Delta e_{ij}^{\left( {k - 1} \right)}` 代表 :math:`t` 时刻（当前）构形下 :math:`t + \Delta t` 时刻 :math:`\Delta e_{ij}` 变量的第:math:`k-1` 个分量。

        ----------------------------------------
        2. 完全的Lagrangian格式有限元方程
        ----------------------------------------

        说明：方程变量分量形式的左下标代表构形，左上标代表时刻，右上标代表哑标，右下标是变量分量指标。
        如 :math:`{}_t^{t + \Delta t}\Delta e_{ij}^{\left( {k - 1} \right)}` 代表 :math:`t` 时刻（当前）构形下 :math:`t + \Delta t` 时刻 :math:`\Delta e_{ij}` 变量的第:math:`k-1` 个分量。

        （1）增量形式的 T.L. 方程：

        .. math::
            \int_{{}^0 V} { }_{0} \mathrm{C}_{i j r s} \ { }_{0} \Delta e_{r s} \delta \ { }_{0} \Delta e_{i j} {}^{0} \mathrm{~d} V +
            \int_{{}^0 V} { }_{0}^{t} S_{i j} \delta \Delta \ { }_{0} \eta_{i j} { }^{0} \mathrm{~d} V ={ }^{t+\Delta t} R -
            \int_{{}^0 V} { }_{0}^{t} S_{i j} \delta \ { }_{0} \Delta e_{i j} { }^{0} \mathrm{~d} V

        （2）采用修正的牛顿迭代求解格式为：

        .. math::
            \int_{{}^0 V} { }_{0} \mathrm{C}_{i j r s} \ {}_{0}\Delta e_{r s}^\left(k \right) \delta \ {}_{0}\Delta e_{i j} \ {}^{0} \! \mathrm{~d} V +
            \int_{{}^0 V} { }_{0}^{t} S_{i j} \delta \ {}_{0} \Delta \eta_{i j}^\left(k \right) \ { }^{0} \! \mathrm{~d} V ={ }^{t+\Delta t} R -
            \int_{{}^0 V} { }_{0}^{t+\Delta t} S_{i j}^\left(k-1 \right) \delta \ {}_{0}^{t+\Delta t} \Delta e_{i j}^\left(k-1 \right) \ { }^{0} \! \mathrm{~d} V

        其中

        .. math::
            {}^{t + \Delta t}R = \int_{{}_0V} {{}_0^{t + \Delta t}} {f_i}\delta {u_i}{\;^0}\;{\text{d}}V +
            \int_{{}_0S} {{}_0^{t + \Delta t}} {t_i}\delta {u_i}{\;^0}\;{\text{d}}S

        （3）T.L. 方程的增量应变记为：

        .. math::
            { }_{0} \Delta \varepsilon_{i j} = \frac{1}{2}\left({ }_{0} \Delta u_{i,j} + \ { }_{0} \Delta u_{j,i} \right) +
            \frac{1}{2} \left( { }_{0}^{t} u_{k,i} \ { }_{0} \Delta u_{k,j}+ \ { }_{0} \Delta u_{k,i} \ { }_{0}^{t} u_{k,j} \right) +
            \frac{1}{2} \left( { }_{0} \Delta u_{k,i} \ { }_{0} \Delta u_{k,j}\right) \;\; (i = 1,2,3,j = 1,2,3,k = 1,2,3)

        （4）相应的计算矩阵有限元离散格式：

        静力分析：

        .. math::
            \left({}_{0}^{t}\left[K_{L}\right]+{ }_{0}^{t}\left[K_{N L}\right] \right) \Delta\{U\}^{(i)} = {}^{t+\Delta t}\{R\} - { }_0^{t+\Delta t}\{F\}^{(i-1)}

        动力分析隐式积分：

        .. math::
            [M] \ {}^{t+\Delta t}\left\{\ddot{U}\right\}^{(i)} + \left({}_{0}^{t}\left[K_{L}\right]+{ }_{0}^{t}\left[K_{N L}\right]\right) \Delta\{U\}^{(i)}
            = {}^{t+\Delta t}\{R\} - \ { }^{t+\Delta t}\{F\}^{(i-1)}

        动力分析显式积分：

        .. math::
            [M] \ {}^{t}\{\ddot U\} = {}^{t}\{R\} - \ {}_0^{t}\{F\}

        其中， :math:`{ }_{0}^{t}\left[K_{L}\right]=\int_{{}^0 V}{ }_{0}^{t}\left[B_{\mathrm{L}}\right]^{\mathrm{T}} \ { }_{0}[C] \ {}_{0}^{t}\left[B_{\mathrm{L}}\right] \ {}^{0} \mathrm{~d} V`
        为线性应变增量刚度矩阵； :math:`{ }_{0}[\mathrm{C}]` 为增量应力一应变材料特性矩阵； :math:`{ }_{0}^{t}\left[B_{\mathrm{L}}\right]` 为线性应变一位移变换矩阵。

        线性应变一位移变换矩阵，使用：

        .. math::
            { }_{0}\{e\}={ }_{0}^{t}\left[B_{L}\right]\{\bar{u}\}

        其中

        .. math::
            _{0}\{e\}^{T}=\left[\begin{array}{llllll}
            { }_{0} e_{11} & { }_{0} e_{22} & { }_{0} e_{33} & 2_{0} e_{12} & 2_{0} e_{13} & 2_{0} e_{23}
            \end{array}\right]

        .. math::
            \{\bar{u}\}^{T}=\left[\begin{array}{llllllllll}
            u_{1}^{1} & u_{2}^{1} & u_{3}^{1} & u_{1}^{2} & u_{2}^{2} & u_{3}^{2} & \cdot \cdot \cdot & u_{1}^{n} & u_{2}^{n} & u_{3}^{n}
            \end{array}\right]

        对于T.L. 方程，线性应变一位移变换矩阵 :math:`{ }_{0}^{t}\left[B_{\mathrm{L}}\right]` 可写为：

        .. math::
            { }_{0}^{t}\left[B_{\mathrm{L}}\right] = { }_{0}^{t}\left[B_{\mathrm{L}_{0}}\right]+{ }_{0}^{t}\left[B_{\mathrm{L}_{1}}\right]

        其中， :math:`{ }_{0}^{t}\left[B_{\mathrm{L}_{0}}\right]` 与一般的线性应变-位移矩阵相同，对应  :math:`\frac{1}{2}\left({ }_{0} \Delta u_{i,j} + { }_{0} \Delta u_{j,i} \right)` 线性应变增量部分

        .. math::
            { }_{0}^{t} \left[B_{L_{0}}\right]=\left(\begin{array}{cccccccc}
            { }_{0} N_{1,1} & 0 & 0 & { }_{0} N_{2,1} & \cdot & \cdot & \cdot & 0 \\
            0 & { }_{0} N_{1,2} & 0 & 0 & \cdot & \cdot & \cdot & 0 \\
            0 & 0 & { }_{0} N_{1,3} & 0 & \cdot & \cdot & \cdot & { }_{0} N_{n, 3} \\
            { }_{0} N_{1,2} & { }_{0} N_{1,1} & 0 & { }_{0} N_{2,2} & \cdot & \cdot & \cdot & 0 \\
            { }_{0} N_{1,3} & 0 & { }_{0} N_{1,1} & { }_{0} N_{2,3} & \cdot & \cdot & \cdot & { }_{0} N_{n, 1} \\
            0 & { }_{0} N_{1,3} & { }_{0} N_{1,2} & 0 & \cdot & \cdot & \cdot & { }_{0} N_{n, 2}
            \end{array}\right)

        其中，

        .. math::
            { }_{0} N_{k, j}=\frac{\partial N_{k}}{\partial^{0} x_{j}}, \; \Delta u_{i}^{k}={ }^{t+\Delta t} u_{i}^{k}- \ { }^{t} u_{i}^{k}

        由初始位移效应引起的应变-位移矩阵 :math:`{ }_{0}^{t}\left[B_{\mathrm{L}_{1}}\right]` ，对应 :math:`\frac{1}{2} \left( { }_{0}^{t} u_{k,i} \ { }_{0} \Delta u_{k,j}+ { }_{0} \Delta u_{k,i} \ { }_{0}^{t} u_{k,j} \right)` 线性应变增量部分：

        .. math::
            { }_{0}^{t} \left[B_{L_{1}}\right]=\left(\begin{array}{ccccccccc}
            l_{11} \ { }_{0} N_{1,1} & l_{21} \ { }_{0} N_{1,1} & l_{31} \ { }_{0} N_{1,1} & l_{11} \ { }_{0} N_{2,1} & \cdot & \cdot & \cdot & \cdot & l_{31} \ { }_{0} N_{n,1} \\
            l_{12} \ { }_{0} N_{1,2} & l_{22} \ { }_{0} N_{1,2} & l_{32} \ { }_{0} N_{1,2} & l_{12} \ { }_{0} N_{2,2} & \cdot & \cdot & \cdot & \cdot & l_{32} \ { }_{0} N_{n,2} \\
            l_{13} \ { }_{0} N_{1,3} & l_{23} \ { }_{0} N_{1,3} & l_{33} \ { }_{0} N_{1,3} & l_{13} \ { }_{0} N_{2,3} & \cdot & \cdot & \cdot & \cdot & l_{33} \ { }_{0} N_{n,3} \\
            a_{11}^{1} & a_{12}^{1} & a_{13}^{1} & a_{11}^{2} & \cdot & \cdot & \cdot & \cdot & a_{13}^{n}
            \\
            a_{31}^{1} & a_{32}^{1} & a_{33}^{1} & a_{31}^{2} & \cdot & \cdot & \cdot & \cdot & a_{33}^{n}
            \\
            a_{21}^{1} & a_{22}^{1} & a_{23}^{1} & a_{21}^{2} & \cdot & \cdot & \cdot & \cdot & a_{23}^{n}
            \end{array}\right)

        其中，

        .. math::
            l_{i j}=\sum_{k=1}^{n} { }_{0} N_{k, j} { }^{t} u_{i}^{k} = \frac{\partial \ { }^{t} u_i}{\partial {}^{0} x_{j}}

        .. math::
            a_{11}^{1}=\left(l_{11} \ { }_{0} N_{1, 2}+l_{12} \ { }_{0}  N_{1, 1}\right) \quad a_{21}^{1}=\left(l_{12} \ { }_{0} N_{1, 3}+l_{13} \ { }_{0}  N_{1, 2}\right) \quad a_{31}^{1}=\left(l_{11} \ { }_{0} N_{1, 3}+l_{13} \ { }_{0}  N_{1, 1}\right)

        .. math::
            a_{12}^{1}=\left(l_{21} \ { }_{0} N_{1, 2}+l_{22} \ { }_{0}  N_{1, 1}\right) \quad a_{22}^{1}=\left(l_{22} \ { }_{0} N_{1, 3}+l_{23} \ { }_{0}  N_{1, 2}\right) \quad a_{32}^{1}=\left(l_{21} \ { }_{0} N_{1, 3}+l_{23} \ { }_{0}  N_{1, 1}\right)

        .. math::
            a_{13}^{1}=\left(l_{31} \ { }_{0} N_{1, 2}+l_{32} \ { }_{0}  N_{1, 1}\right) \quad a_{23}^{1}=\left(l_{32} \ { }_{0} N_{1, 3}+l_{33} \ { }_{0}  N_{1, 2}\right) \quad a_{33}^{1}=\left(l_{31} \ { }_{0} N_{1, 2}+l_{33} \ { }_{0}  N_{1, 1}\right)

        .. math::
            a_{11}^{2}=\left(l_{11} \ { }_{0} N_{2, 2}+l_{12} \ { }_{0}  N_{2, 1}\right) \quad a_{21}^{2}=\left(l_{12} \ { }_{0} N_{2, 3}+l_{13} \ { }_{0}  N_{2, 2}\right) \quad a_{31}^{2}=\left(l_{11} \ { }_{0} N_{2, 3}+l_{13} \ { }_{0}  N_{2, 1}\right)

        .. math::
            a_{13}^{n}=\left(l_{31} \ { }_{0} N_{n, 2}+l_{32} \ { }_{0}  N_{n, 1}\right) \quad a_{23}^{n}=\left(l_{32} \ { }_{0} N_{n, 3}+l_{33} \ { }_{0}  N_{n, 2}\right) \quad a_{33}^{n}=\left(l_{31} \ { }_{0} N_{n, 2}+l_{33} \ { }_{0}  N_{n, 1}\right)

        整个线性应变一位移变换矩阵 :math:`{ }_{0}^{t}\left[B_{\mathrm{L}}\right]={ }_{0}^{t}\left[B_{\mathrm{L}_{0}}\right]+{ }_{0}^{t}\left[B_{\mathrm{L}_{1}}\right]` 整理为：

        .. math::
            { }_{0}^{t} \left[B_{L}\right]=\left(\begin{array}{ccccccccc}
            { }_{0} N_{1,1} + l_{11} \ { }_{0} N_{1,1} & l_{21} \ { }_{0} N_{1,1} & l_{31} \ { }_{0} N_{1,1} & { }_{0} N_{2,1} +  l_{11} \ { }_{0} N_{2,1} & \cdot & \cdot & \cdot & \cdot & l_{31} \ { }_{0} N_{n,1} \\
            l_{12} \ { }_{0} N_{1,2} & { }_{0} N_{1,2} +  l_{22} \ { }_{0} N_{1,2} & l_{32} \ { }_{0} N_{1,2} & l_{12} \ { }_{0} N_{2,2} & \cdot & \cdot & \cdot & \cdot & l_{32} \ { }_{0} N_{n,2} \\
            l_{13} \ { }_{0} N_{1,3} & l_{23} \ { }_{0} N_{1,3} & { }_{0} N_{1,3} +  l_{33} \ { }_{0} N_{1,3} & l_{13} \ { }_{0} N_{2,3} & \cdot & \cdot & \cdot & \cdot & { }_{0} N_{n,3} + l_{33} \ { }_{0} N_{n,3} \\
            { }_{0} N_{1,2} + \left(l_{11} \ { }_{0} N_{1, 2}+l_{12} \ { }_{0}  N_{1, 1}\right) & { }_{0} N_{1,1} +  \left(l_{21} \ { }_{0} N_{1, 2}+l_{22} \ { }_{0}  N_{1, 1}\right) & \left(l_{31} \ { }_{0} N_{1, 2}+l_{32} \ { }_{0}  N_{1, 1}\right) &{ }_{0} N_{1,2} +  \left(l_{11} \ { }_{0} N_{2, 2}+l_{12} \ { }_{0}  N_{2, 1}\right) & \cdot & \cdot & \cdot & \cdot & \left(l_{31} \ { }_{0} N_{n, 2}+l_{32} \ { }_{0}  N_{n, 1}\right)
            \\
            { }_{0} N_{1,3} + \left(l_{11} \ { }_{0} N_{1, 3}+l_{13} \ { }_{0}  N_{1, 1}\right) & \left(l_{21} \ { }_{0} N_{1, 3}+l_{23} \ { }_{0}  N_{1, 1}\right) &{ }_{0} N_{1,1} + \left(l_{31} \ { }_{0} N_{1, 2}+l_{33} \ { }_{0}  N_{1, 1}\right) &{ }_{0} N_{2,3} +  \left(l_{11} \ { }_{0} N_{2, 3}+l_{13} \ { }_{0}  N_{2, 1}\right) & \cdot & \cdot & \cdot & \cdot &{ }_{0} N_{n,1} +  \left(l_{31} \ { }_{0} N_{n, 2}+l_{33} \ { }_{0}  N_{n, 1}\right)
            \\
            \left(l_{12} \ { }_{0} N_{1, 3}+l_{13} \ { }_{0}  N_{1, 2}\right) &{ }_{0} N_{1,3} + \left(l_{22} \ { }_{0} N_{1, 3}+l_{23} \ { }_{0}  N_{1, 2}\right) &{ }_{0} N_{1,2} +  \left(l_{32} \ { }_{0} N_{1, 3}+l_{33} \ { }_{0}  N_{1, 2}\right) & \left(l_{12} \ { }_{0} N_{2, 3}+l_{13} \ { }_{0}  N_{2, 2}\right) & \cdot & \cdot & \cdot & \cdot &{ }_{0} N_{n,2} +  \left(l_{32} \ { }_{0} N_{n, 3}+l_{33} \ { }_{0}  N_{n, 2}\right)
            \end{array}\right)

        化简得到：

        .. math::
            { }_{0}^{t} \left[B_{L}\right]=\left(\begin{array}{ccccccccc}
            F_{11} \ { }_{0} N_{1,1} & F_{21} \ { }_{0} N_{1,1} & F_{31} \ { }_{0} N_{1,1} & F_{11} \ { }_{0} N_{2,1}  & \cdot & \cdot & \cdot & F_{31} \ { }_{0} N_{n,1} \\
            F_{12} \ { }_{0} N_{1,2} & F_{22} \ { }_{0} N_{1,2} & F_{32} \ { }_{0} N_{1,2} & F_{12} \ { }_{0} N_{2,2}  & \cdot & \cdot & \cdot & F_{32} \ { }_{0} N_{n,2} \\
            F_{13} \ { }_{0} N_{1,3} & F_{23} \ { }_{0} N_{1,3} & F_{33} \ { }_{0} N_{1,3} & l_{13} \ { }_{0} N_{2,3}  & \cdot & \cdot & \cdot & F_{33} \ { }_{0} N_{n,3} \\
            F_{11} \ { }_{0} N_{1, 2}+F_{12} \ { }_{0}  N_{1, 1} & F_{21} \ { }_{0} N_{1, 2}+F_{22} \ { }_{0}  N_{1, 1} & F_{31} \ { }_{0} N_{1, 2}+F_{32} \ { }_{0}  N_{1, 1} & F_{11} \ { }_{0} N_{2, 2}+F_{12} \ { }_{0}  N_{2, 1} & \cdot & \cdot & \cdot & F_{31} \ { }_{0} N_{n, 2}+F_{32} \ { }_{0}  N_{n, 1}
            \\
            F_{11} \ { }_{0} N_{1, 3}+F_{13} \ { }_{0}  N_{1, 1} & F_{21} \ { }_{0} N_{1, 3}+F_{23} \ { }_{0}  N_{1, 1} & F_{31} \ { }_{0} N_{1, 2}+F_{33} \ { }_{0}  N_{1, 1} & F_{11} \ { }_{0} N_{2, 3}+F_{13} \ { }_{0}  N_{2, 1} & \cdot & \cdot & \cdot & F_{31} \ { }_{0} N_{n, 2}+F_{33} \ { }_{0}  N_{n, 1}
            \\
            F_{12} \ { }_{0} N_{1, 3}+F_{13} \ { }_{0}  N_{1, 2} & F_{22} \ { }_{0} N_{1, 3}+F_{23} \ { }_{0}  N_{1, 2} & F_{32} \ { }_{0} N_{1, 3}+F_{33} \ { }_{0}  N_{1, 2} & F_{12} \ { }_{0} N_{2, 3}+F_{13} \ { }_{0}  N_{2, 2} & \cdot & \cdot & \cdot & F_{32} \ { }_{0} N_{n, 3}+F_{33} \ { }_{0}  N_{n, 2}
            \end{array}\right)


        这里， :math:`{ }_{0}^t[S]` 为第二Kirchhoff应力矩阵；

        .. math::
            { }_{0}^{t}[S]=\left(\begin{array}{ccc}
            { }_{0}^{t}[\bar{S}] & {[\bar{O}]} & {[\bar{O}]} \\
            {[\bar{O}]} & { }_{0}^{t}[\bar{S}] & {[\bar{O}]} \\
            {[\bar{O}]} & {[\bar{O}]} & { }_{0}^{t} [\bar{S}]
            \end{array}\right), \quad[\bar{O}]=\left(\begin{array}{lll}
            0 & 0 & 0 \\
            0 & 0 & 0 \\
            0 & 0 & 0
            \end{array}\right)

        其中，

        .. math::
            { }_{0}^{t}[\bar S]=\left(\begin{array}{ccc}
            { }_{0}^{t} S_{11} & { }_{0}^{t} S_{12} & { }_{0}^{t} S_{13} \\
            { }_{0}^{t} S_{21} & { }_{0}^{t} S_{22} & { }_{0}^{t} S_{23} \\
            { }_{0}^{t} S_{31} & { }_{0}^{t} S_{32} & { }_{0}^{t} S_{33}
            \end{array}\right)

        其中， :math:`{ }_{0}^{t}\{F\} = \int_{{}^0 V}{ }_{0}^{t}\left[B_{\mathrm{L}}\right]^{\mathrm{T}} \ { }_{0}^{t}[\bar S] \ {}^{0} \mathrm{~d} V`
        为 :math:`t` 时刻的单元应力的等效结点力矢量。

        其中， :math:`{}^{t}\left\{\ddot{U}\right\}` 为 :math:`t` 时刻的结点加速度矢量。

        其中， :math:`[M]=\int_{{}^0 V}[N]^{T}[N] \ {}^{0} \mathrm{~d} V` 为与时间无关的质量矩阵。

        其中， :math:`{}^{t+\Delta t}\left\{\ddot{U}\right\}^{(i)}` 为 :math:`t+\Delta t` 时刻对应于第 :math:`i` 次迭代的结点加速度矢量。

        --------------------------------------------------
        3. 更新的 Lagrangian 格式有限元方程
        --------------------------------------------------

        （1）增量形式的 U.L. 方程：

        .. math::
            \int_{{}^t V} {{}_t} {C_{ijrs}}\Delta {e_{rs}}\delta \;\Delta {e_{ij}}^t\;{\text{d}}V + \int_{{}^t V} {{}^t}
            {\sigma_{ij}}\delta \;\Delta {\eta_{ij}}^t\;{\text{d}}V{ = {}^{t + \Delta t}}R - \int_{{}^t V} {{}^t}
            {\sigma _{ij}}\delta \;\Delta {e_{ij}}^t\;{\text{d}}V

        （2）采用修正的牛顿迭代求解格式为：

        .. math::
            \int_{{}^t V} {{}_t} {{\text{C}}_{ijrs}}{{}_t}\Delta e_{rs}^{\left( k \right)}\delta {{}_t}\Delta {e_{ij}}
            {{}^t}\;{\text{d}}V + \int_{{}^t V} {{}^t} {\sigma _{ij}}\delta {{}_t}\Delta \eta_{ij}^{\left( k \right)}
            {{}^t}{\text{d}}V{ = {}^{t + \Delta t}}R - \int_{{}^t V} {{}^{t + \Delta t}} \sigma _{ij}^{\left( {k - 1}
            \right)}\delta {}_t^{t + \Delta t}\Delta e_{ij}^{\left( {k - 1} \right)}{{}^{t + \Delta t}}\;{\text{d}}V

        其中

        .. math::
            {}^{t + \Delta t}R = \int_{{}_0V} {{}_0^{t + \Delta t}} {f_i}\delta {u_i}{{}^0}\;{\text{d}}V + \int_{{}_0 S}
            {{}_0^{t + \Delta t}} {t_i}\delta {u_i}{{}^0}\;{\text{d}}S

        （3）U.L. 方程的增量应变记为：

        .. math::
            {}_t\Delta {\varepsilon_{ij}} = \frac{1}{2}\left( {{}_t\Delta {u_{i,j}}+ \ {}_t \Delta {u_{j,i}}} \right) +
            + \frac{1}{2}{\;_t}\Delta {u_{k,i}}{\;_t}\Delta {u_{k,j}}\;\;\;(i = 1,2,3,j = 1,2,3,k = 1,2,3)

        （4）相应的计算矩阵有限元离散格式：

        静力分析：

        .. math::
            \left( {{}_t^t\left[ {{K_L}} \right] + {}_t^t\left[ {{K_{NL}}} \right]} \right)\Delta {\{ U\}^{(i)}} =
            {}^{t + \Delta t}\{ R\}  - {}_t^{t + \Delta t}{\{ F\} ^{(i - 1)}}

        动力分析隐式积分：

        .. math::
            [M]{{}^{t + \Delta t}}{\left\{ {\ddot U} \right\}^{(i)}} + \left( {{}_t^t\left[ {{K_L}} \right] +
            {}_t^t\left[ {{K_{NL}}} \right]} \right)\Delta {\{ U\}^{(i)}}{ = {}^{t + \Delta t}}\{ R\}  - {}_t^{t + \Delta t}{\{ F\} ^{(i - 1)}}

        动力分析显式积分：

        .. math::
            [M]{{}^t}\{ \ddot U\} { = {}^t}\{ R\}  - {}_t^t\{ F\}

        其中， :math:`{ }_{t}^{t}\left[K_{L}\right]=\int_{{}^t V}{ }_{t}^{t}\left[B_{\mathrm{L}}\right]^{\mathrm{T}} \ { }_{t}[C] \ {}_{t}^{t}\left[B_{\mathrm{L}}\right] \ {}^{t} \mathrm{~d} V`
        为线性应变增量刚度矩阵； :math:`{ }_{t}[\mathrm{C}]` 为增量应力一应变材料特性矩阵； :math:`{ }_{t}^{t}\left[B_{\mathrm{L}}\right]` 为线性应变一位移变换矩阵。

        线性应变一位移变换矩阵，使用：

        .. math::
            { }_{t}\{e\}={ }_{t}^{t}\left[B_{L}\right]\{\bar{u}\}

        其中

        .. math::
            { }_{t}\{e\}^{T}=\left[\begin{array}{llllll}
            { }_{t} e_{11} & { }_{t} e_{22} & { }_{t} e_{33} & 2 \ { }_{t} e_{12} & 2 \ { }_{t} e_{13} & 2 \ { }_{t} e_{23}
            \end{array}\right]

        .. math::
            \{\bar{u}\}^{T}=\left[\begin{array}{llllllllll}
            u_{1}^{1} & u_{2}^{1} & u_{3}^{1} & u_{1}^{2} & u_{2}^{2} & u_{3}^{2} & \cdot \cdot \cdot & u_{1}^{n} & u_{2}^{n} & u_{3}^{n}
            \end{array}\right]

        对于 U.L. 方程， :math:`{ }_{t}^{t}\left[B_{\mathrm{L}}\right]` 与一般的线性应变-位移矩阵相同，对应  :math:`\frac{1}{2}\left({ }_{t} \Delta u_{i,j} + { }_{t} \Delta u_{j,i} \right)` 线性应变增量部分

        .. math::
            { }_{t}^{t} \left[B_{L}\right]=\left(\begin{array}{cccccccc}
            { }_{t} N_{1,1} & 0 & 0 & { }_{t} N_{2,1} & \cdot & \cdot & \cdot & 0 \\
            0 & { }_{t} N_{1,2} & 0 & 0 & \cdot & \cdot & \cdot & 0 \\
            0 & 0 & { }_{t} N_{1,3} & 0 & \cdot & \cdot & \cdot & { }_{t} N_{n, 3} \\
            { }_{t} N_{1,2} & { }_{t} N_{1,1} & 0 & { }_{t} N_{2,2} & \cdot & \cdot & \cdot & 0 \\
            { }_{t} N_{1,3} & 0 & { }_{t} N_{1,1} & { }_{t} N_{2,3} & \cdot & \cdot & \cdot & { }_{t} N_{n, 1} \\
            0 & { }_{t} N_{1,3} & { }_{t} N_{1,2} & 0 & \cdot & \cdot & \cdot & { }_{t} N_{n, 2}
            \end{array}\right)

        其中，

        .. math::
            { }_{t} N_{k, j}=\frac{\partial N_{k}}{\partial^{t} x_{j}}, \; \Delta u_{i}^{k}= \ { }^{t+\Delta t} u_{i}^{k}- \ { }^{t} u_{i}^{k}

        这里，Cauchy 应力矩阵 :math:`{ }^t[\sigma]` 表示为:

        .. math::
            { }^{t}[\sigma]=\left(\begin{array}{ccc}
            { }^{t}[\bar{\sigma}] & {[\bar{O}]} & {[\bar{O}]} \\
            {[\bar{O}]} & { }^{t}[\bar{\sigma}] & {[\bar{O}]} \\
            {[\bar{O}]} & {[\bar{O}]} & { }^{t} [\bar{\sigma}]
            \end{array}\right), \quad[\bar{O}]=\left(\begin{array}{lll}
            0 & 0 & 0 \\
            0 & 0 & 0 \\
            0 & 0 & 0
            \end{array}\right)

        其中，

        .. math::
            { }^{t}[\bar \sigma]=\left(\begin{array}{ccc}
            { }^{t} \sigma_{11} & { }^{t} \sigma_{12} & { }^{t} \sigma_{13} \\
            { }^{t} \sigma_{21} & { }^{t} \sigma_{22} & { }^{t} \sigma_{23} \\
            { }^{t} \sigma_{31} & { }^{t} \sigma_{32} & { }^{t} \sigma_{33}
            \end{array}\right)

        其中， :math:`{ }_{t}^{t}\{F\} = \int_{{}^t V}{ }_{t}^{t}\left[B_{\mathrm{L}}\right]^{\mathrm{T}} \ { }^t [\sigma] \ {}^{t} \mathrm{~d} V`
        为 :math:`t` 时刻的单元应力的等效结点力矢量。

        其中， :math:`{}^{t}\left\{\ddot{U}\right\}` 为 :math:`t` 时刻的结点加速度矢量。

        其中， :math:`[M]=\int_{{}^0 V}[N]^{T}[N] \ {}^{0} \mathrm{~d} V` 为与时间无关的质量矩阵。

        其中， :math:`{}^{t+\Delta t}\left\{\ddot{U}\right\}^{(i)}` 为 :math:`t+\Delta t` 时刻对应于第 :math:`i` 次迭代的结点加速度矢量。

        注意：本程序中使用的应力矢量Vogit记法为：

        .. math::
            { }_{0}^{t} \{ \bar \sigma \}= [{ }_{0}^{t} \sigma_{11} \ { }_{0}^{t} \sigma_{22}  \  { }_{0}^{t} \sigma_{33} \  { }_{0}^{t} \sigma_{12}  \  { }_{0}^{t} \sigma_{13}  \  { }_{0}^{t} \sigma_{23}]^{T}

        因此，对于三维问题，上述推导中的线性应变矩阵与非线性应变矩阵的最后两行已经进行了交换，得到本程序计算出的结果。

        参考文献：

        [1] 连续体和结构的非线性有限元_Ted Belytschko等著 庄茁译_2002
        [2] 非线性有限元分析-张汝清-1990
        [3] Non-linear finite element analysis of solids and structures. 2012.

        """
        if self.dimension == 2:
            self.qp_b_matrices = np.zeros(shape=(self.qp_number, 3, self.element_dof_number), dtype=DTYPE)
        elif self.dimension == 3:
            self.qp_b_matrices = np.zeros(shape=(self.qp_number, 6, self.element_dof_number), dtype=DTYPE)

        if self.method == "TL":
            for iqp, (qp_shape_gradient, F1, qp_jacobi_inv) in enumerate(zip(self.iso_element_shape.qp_shape_gradients,
                                                                             self.qp_deformation_gradients_1, self.qp_jacobi_invs)):
                qp_dhdx = np.dot(qp_shape_gradient.transpose(), qp_jacobi_inv)
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
                        self.qp_b_matrices[iqp, 3, i * 3 + 2] = val[1] * F1[2, 0] + val[0] * F1[2, 1]
                        self.qp_b_matrices[iqp, 4, i * 3 + 0] = val[0] * F1[0, 2] + val[2] * F1[0, 0]
                        self.qp_b_matrices[iqp, 4, i * 3 + 1] = val[0] * F1[1, 2] + val[2] * F1[1, 0]
                        self.qp_b_matrices[iqp, 4, i * 3 + 2] = val[0] * F1[2, 2] + val[2] * F1[2, 0]

                        self.qp_b_matrices[iqp, 5, i * 3 + 0] = val[2] * F1[0, 1] + val[1] * F1[0, 2]
                        self.qp_b_matrices[iqp, 5, i * 3 + 1] = val[2] * F1[1, 1] + val[1] * F1[1, 2]
                        self.qp_b_matrices[iqp, 5, i * 3 + 2] = val[2] * F1[2, 1] + val[1] * F1[2, 2]

        elif self.method == "UL":
            self.cal_jacobi_t()
            for iqp, (qp_shape_gradient, qp_jacobi_inv_t) in enumerate(zip(self.iso_element_shape.qp_shape_gradients, self.qp_jacobi_invs_t)):
                qp_dhdx_t = np.dot(qp_shape_gradient.transpose(), qp_jacobi_inv_t)
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

        self.qp_b_matrices_transpose = np.array([qp_b_matrix.transpose() for qp_b_matrix in self.qp_b_matrices])

    def create_qp_bnl_matrices(self) -> None:
        r"""
        **获得 Lagrangian 网格的非线性应变-位移矩阵**

        单元使用的节点坐标和位移插值是：

        .. math::
            {}^0{x_i} = \sum\limits_{k = 1}^n {{N_k}} {\;^0}x_i^k,\; {{}^t}{x_i} = \sum\limits_{k = 1}^n {{N_k}} {\;^t}x_i^k \;\;(i = 1,2,3)

        .. math::
            {}^t{u_i} = \sum\limits_{k = 1}^n {{N_k}} {\;^t}u_i^k,\; \Delta {u_i} = \sum\limits_{k = 1}^n {{N_k}} \;\Delta u_i^k \;\;(i = 1,2,3)

        这里， :math:`N_{k}` 是单元插值函数， :math:`k` 是节点个数。现推导 T. L. 和 U. L. 公式中有关单元的矩阵。

        ----------------------------------------
        1. 完全的Lagrangian格式有限元方程
        ----------------------------------------

        说明：方程变量分量形式的左下标代表构形，左上标代表时刻，右上标代表哑标，右下标是变量分量指标。
        如 :math:`{}_t^{t + \Delta t}\Delta e_{ij}^{\left( {k - 1} \right)}` 代表 :math:`t` 时刻（当前）构形下 :math:`t + \Delta t` 时刻 :math:`\Delta e_{ij}` 变量的第:math:`k-1` 个分量。

        （1）增量形式的 T.L. 方程：

        .. math::
            \int_{{}^0 V} { }_{0} \mathrm{C}_{i j r s} \ { }_{0} \Delta e_{r s} \delta \ { }_{0} \Delta e_{i j} {}^{0} \mathrm{~d} V +
            \int_{{}^0 V} { }_{0}^{t} S_{i j} \delta \Delta \ { }_{0} \eta_{i j} { }^{0} \mathrm{~d} V ={ }^{t+\Delta t} R -
            \int_{{}^0 V} { }_{0}^{t} S_{i j} \delta \ { }_{0} \Delta e_{i j} { }^{0} \mathrm{~d} V

        （2）采用修正的牛顿迭代求解格式为：

        .. math::
            \int_{{}^0 V} { }_{0} \mathrm{C}_{i j r s} \ {}_{0}\Delta e_{r s}^\left(k \right) \delta \ {}_{0}\Delta e_{i j} \ {}^{0} \! \mathrm{~d} V +
            \int_{{}^0 V} { }_{0}^{t} S_{i j} \delta \ {}_{0} \Delta \eta_{i j}^\left(k \right) \ { }^{0} \! \mathrm{~d} V ={ }^{t+\Delta t} R -
            \int_{{}^0 V} { }_{0}^{t+\Delta t} S_{i j}^\left(k-1 \right) \delta \ {}_{0}^{t+\Delta t} \Delta e_{i j}^\left(k-1 \right) \ { }^{0} \! \mathrm{~d} V

        其中

        .. math::
            {}^{t + \Delta t}R = \int_{{}_0V} {{}_0^{t + \Delta t}} {f_i}\delta {u_i}{\;^0}\;{\text{d}}V +
            \int_{{}_0S} {{}_0^{t + \Delta t}} {t_i}\delta {u_i}{\;^0}\;{\text{d}}S

        （3）T.L. 方程的增量应变记为：

        .. math::
            { }_{0} \Delta \varepsilon_{i j} = \frac{1}{2}\left({ }_{0} \Delta u_{i,j} + \ { }_{0} \Delta u_{j,i} \right) +
            \frac{1}{2} \left( { }_{0}^{t} u_{k,i} \ { }_{0} \Delta u_{k,j}+ \ { }_{0} \Delta u_{k,i} \ { }_{0}^{t} u_{k,j} \right) +
            \frac{1}{2} \left( { }_{0} \Delta u_{k,i} \ { }_{0} \Delta u_{k,j}\right) \;\; (i = 1,2,3,j = 1,2,3,k = 1,2,3)

        （4）相应的计算矩阵有限元离散格式：

        静力分析：

        .. math::
            \left({}_{0}^{t}\left[K_{L}\right]+{ }_{0}^{t}\left[K_{N L}\right] \right) \Delta\{U\}^{(i)} = {}^{t+\Delta t}\{R\} - { }_0^{t+\Delta t}\{F\}^{(i-1)}

        动力分析隐式积分：

        .. math::
            [M] \ {}^{t+\Delta t}\left\{\ddot{U}\right\}^{(i)} + \left({}_{0}^{t}\left[K_{L}\right]+{ }_{0}^{t}\left[K_{N L}\right]\right) \Delta\{U\}^{(i)}
            = {}^{t+\Delta t}\{R\} - \ { }^{t+\Delta t}\{F\}^{(i-1)}

        动力分析显式积分：

        .. math::
            [M] \ {}^{t}\{\ddot U\} = {}^{t}\{R\} - \ {}_0^{t}\{F\}

        其中， :math:`{ }_{0}^{t}\left[K_{NL}\right]=\int_{{}^0 V}{ }_{0}^{t}\left[B_{\mathrm{NL}}\right]^{\mathrm{T}} \ { }_{0}^t[S] \ {}_{0}^{t}\left[B_{\mathrm{NL}}\right] \ {}^{0} \mathrm{~d} V`
        为非线性应变(几何或初始应力) 增量刚度矩阵； :math:`{ }_{0}^{t} \left[B_{\mathrm{N} L}\right]` 为非线性应变一位移变换矩阵。

        .. math::
            { }_{0}^{t}\left[B_{N L}\right]=\left(\begin{array}{ccc}
            { }_{0}^{t}\left[\bar{B}_{N L}\right] & {[\bar{O}]} & {[\bar{O}]} \\
            {[\bar{O}]} & { }_{0}^{t}\left[\bar{B}_{N L}\right] & {[\bar{O}]} \\
            {[\bar{O}]} & {[\bar{O}]} & { }_{0}^{t} \left[\bar{B}_{N L}\right]
            \end{array}\right), \quad[\bar {O}]=\left(\begin{array}{c}
            0 \\
            0 \\
            0
            \end{array}\right)

        其中，

        .. math::
            { }_{0}^{t}\left[\bar{B}_{N L}\right]=\left(\begin{array}{cccccccc}
            { }_{0} N_{1,1} & 0 & 0 & { }_{0} N_{2,1} & \cdot & \cdot & \cdot & { }_{0} N_{n, 1} \\
            { }_{0} N_{1,2} & 0 & 0 & { }_{0} N_{2,2} & \cdot & \cdot & \cdot & { }_{0} N_{n, 2} \\
            { }_{0} N_{1,3} & 0 & 0 & { }_{0} N_{2,3} & \cdot & \cdot & \cdot & { }_{0} N_{n, 3}
            \end{array}\right)

        这里， :math:`{ }_{0}^t[S]` 为第二Kirchhoff应力矩阵；

        .. math::
            { }_{0}^{t}[S]=\left(\begin{array}{ccc}
            { }_{0}^{t}[\bar{S}] & {[\bar{O}]} & {[\bar{O}]} \\
            {[\bar{O}]} & { }_{0}^{t}[\bar{S}] & {[\bar{O}]} \\
            {[\bar{O}]} & {[\bar{O}]} & { }_{0}^{t} [\bar{S}]
            \end{array}\right), \quad[\bar{O}]=\left(\begin{array}{lll}
            0 & 0 & 0 \\
            0 & 0 & 0 \\
            0 & 0 & 0
            \end{array}\right)

        其中，

        .. math::
            { }_{0}^{t}[\bar S]=\left(\begin{array}{ccc}
            { }_{0}^{t} S_{11} & { }_{0}^{t} S_{12} & { }_{0}^{t} S_{13} \\
            { }_{0}^{t} S_{21} & { }_{0}^{t} S_{22} & { }_{0}^{t} S_{23} \\
            { }_{0}^{t} S_{31} & { }_{0}^{t} S_{32} & { }_{0}^{t} S_{33}
            \end{array}\right)

        其中， :math:`{ }_{0}^{t}\{F\} = \int_{{}^0 V}{ }_{0}^{t}\left[B_{\mathrm{L}}\right]^{\mathrm{T}} \ { }_{0}^{t}[\bar S] \ {}^{0} \mathrm{~d} V`
        为 :math:`t` 时刻的单元应力的等效结点力矢量。

        其中， :math:`{}^{t}\left\{\ddot{U}\right\}` 为 :math:`t` 时刻的结点加速度矢量。

        其中， :math:`[M]=\int_{{}^0 V}[N]^{T}[N] \ {}^{0} \mathrm{~d} V` 为与时间无关的质量矩阵。

        其中， :math:`{}^{t+\Delta t}\left\{\ddot{U}\right\}^{(i)}` 为 :math:`t+\Delta t` 时刻对应于第 :math:`i` 次迭代的结点加速度矢量。

        ----------------------------------------
        2. 更新的 Lagrangian 格式有限元方程
        ----------------------------------------

        （1）增量形式的 U.L. 方程：

        .. math::
            \int_{{}^t V} {{}_t} {C_{ijrs}}\Delta {e_{rs}}\delta \;\Delta {e_{ij}}^t\;{\text{d}}V + \int_{{}^t V} {{}^t}
            {\sigma_{ij}}\delta \;\Delta {\eta_{ij}}^t\;{\text{d}}V{ = {}^{t + \Delta t}}R - \int_{{}^t V} {{}^t}
            {\sigma _{ij}}\delta \;\Delta {e_{ij}}^t\;{\text{d}}V

        （2）采用修正的牛顿迭代求解格式为：

        .. math::
            \int_{{}^t V} {{}_t} {{\text{C}}_{ijrs}}{{}_t}\Delta e_{rs}^{\left( k \right)}\delta {{}_t}\Delta {e_{ij}}
            {{}^t}\;{\text{d}}V + \int_{{}^t V} {{}^t} {\sigma _{ij}}\delta {{}_t}\Delta \eta_{ij}^{\left( k \right)}
            {{}^t}{\text{d}}V{ = {}^{t + \Delta t}}R - \int_{{}^t V} {{}^{t + \Delta t}} \sigma _{ij}^{\left( {k - 1}
            \right)}\delta {}_t^{t + \Delta t}\Delta e_{ij}^{\left( {k - 1} \right)}{{}^{t + \Delta t}}\;{\text{d}}V

        其中

        .. math::
            {}^{t + \Delta t}R = \int_{{}_0V} {{}_0^{t + \Delta t}} {f_i}\delta {u_i}{{}^0}\;{\text{d}}V + \int_{{}_0 S}
            {{}_0^{t + \Delta t}} {t_i}\delta {u_i}{{}^0}\;{\text{d}}S

        （3）U.L. 方程的增量应变记为：

        .. math::
            {}_t\Delta {\varepsilon_{ij}} = \frac{1}{2}\left( {{}_t\Delta {u_{i,j}}+ \ {}_t \Delta {u_{j,i}}} \right) +
            + \frac{1}{2}{\;_t}\Delta {u_{k,i}}{\;_t}\Delta {u_{k,j}}\;\;\;(i = 1,2,3,j = 1,2,3,k = 1,2,3)

        （4）相应的计算矩阵有限元离散格式：

        静力分析：

        .. math::
            \left( {{}_t^t\left[ {{K_L}} \right] + {}_t^t\left[ {{K_{NL}}} \right]} \right)\Delta {\{ U\}^{(i)}} =
            {}^{t + \Delta t}\{ R\}  - {}_t^{t + \Delta t}{\{ F\} ^{(i - 1)}}

        动力分析隐式积分：

        .. math::
            [M]{{}^{t + \Delta t}}{\left\{ {\ddot U} \right\}^{(i)}} + \left( {{}_t^t\left[ {{K_L}} \right] +
            {}_t^t\left[ {{K_{NL}}} \right]} \right)\Delta {\{ U\}^{(i)}}{ = {}^{t + \Delta t}}\{ R\}  - {}_t^{t + \Delta t}{\{ F\} ^{(i - 1)}}

        动力分析显式积分：

        .. math::
            [M]{{}^t}\{ \ddot U\} { = {}^t}\{ R\}  - {}_t^t\{ F\}

        其中， :math:`{ }_{t}^{t}\left[K_{NL}\right]=\int_{{}^t V}{ }_{t}^{t}\left[B_{\mathrm{NL}}\right]^{\mathrm{T}} \ { }^t[\sigma] \ {}_{t}^{t}\left[B_{\mathrm{NL}}\right] \ {}^{t} \mathrm{~d} V`
        为非线性应变(几何或初始应力) 增量刚度矩阵； :math:`{ }^t[\sigma]` 为 Cauchy 应力矩阵； :math:`{ }_{t}^{t} \left[B_{\mathrm{N} L}\right]` 为非线性应变一位移变换矩阵。

        .. math::
            { }_{t}^{t}\left[B_{N L}\right]=\left(\begin{array}{ccc}
            { }_{t}^{t}\left[\bar{B}_{N L}\right] & {[\bar{O}]} & {[\bar{O}]} \\
            {[\bar{O}]} & { }_{t}^{t}\left[\bar{B}_{N L}\right] & {[\bar{O}]} \\
            {[\bar{O}]} & {[\bar{O}]} & { }_{t}^{t} \left[\bar{B}_{N L}\right]
            \end{array}\right), \quad[\bar {O}]=\left(\begin{array}{c}
            0 \\
            0 \\
            0
            \end{array}\right)

        其中，

        .. math::
            { }_{t}^{t}\left[\bar{B}_{N L}\right]=\left(\begin{array}{cccccccc}
            { }_{t} N_{1,1} & 0 & 0 & { }_{t} N_{2,1} & \cdot & \cdot & \cdot & { }_{t} N_{n, 1} \\
            { }_{t} N_{1,2} & 0 & 0 & { }_{t} N_{2,2} & \cdot & \cdot & \cdot & { }_{t} N_{n, 2} \\
            { }_{t} N_{1,3} & 0 & 0 & { }_{t} N_{2,3} & \cdot & \cdot & \cdot & { }_{t} N_{n, 3}
            \end{array}\right)

        这里，Cauchy 应力矩阵 :math:`{ }^t[\sigma]` 表示为:

        .. math::
            { }^{t}[\sigma]=\left(\begin{array}{ccc}
            { }^{t}[\bar{\sigma}] & {[\bar{O}]} & {[\bar{O}]} \\
            {[\bar{O}]} & { }^{t}[\bar{\sigma}] & {[\bar{O}]} \\
            {[\bar{O}]} & {[\bar{O}]} & { }^{t} [\bar{\sigma}]
            \end{array}\right), \quad[\bar{O}]=\left(\begin{array}{lll}
            0 & 0 & 0 \\
            0 & 0 & 0 \\
            0 & 0 & 0
            \end{array}\right)

        其中，

        .. math::
            { }^{t}[\bar \sigma]=\left(\begin{array}{ccc}
            { }^{t} \sigma_{11} & { }^{t} \sigma_{12} & { }^{t} \sigma_{13} \\
            { }^{t} \sigma_{21} & { }^{t} \sigma_{22} & { }^{t} \sigma_{23} \\
            { }^{t} \sigma_{31} & { }^{t} \sigma_{32} & { }^{t} \sigma_{33}
            \end{array}\right)

        其中， :math:`{ }_{t}^{t}\{F\} = \int_{{}^t V}{ }_{t}^{t}\left[B_{\mathrm{L}}\right]^{\mathrm{T}} \ { }^t [\sigma] \ {}^{t} \mathrm{~d} V`
        为 :math:`t` 时刻的单元应力的等效结点力矢量。

        其中， :math:`{}^{t}\left\{\ddot{U}\right\}` 为 :math:`t` 时刻的结点加速度矢量。

        其中， :math:`[M]=\int_{{}^0 V}[N]^{T}[N] \ {}^{0} \mathrm{~d} V` 为与时间无关的质量矩阵。

        其中， :math:`{}^{t+\Delta t}\left\{\ddot{U}\right\}^{(i)}` 为 :math:`t+\Delta t` 时刻对应于第 :math:`i` 次迭代的结点加速度矢量。

        注意：上述推导中，使用的应力矢量Vogit记法为：

        .. math::
            { }_{0}^{t} \{ \bar S \}= [{ }_{0}^{t} S_{11} \ { }_{0}^{t} S_{22}  \  { }_{0}^{t} S_{33} \  { }_{0}^{t} S_{12}  \  { }_{0}^{t} S_{13}  \  { }_{0}^{t} S_{23}]^{T}

        因此，对于三维问题，上述推导中的线性应变矩阵与非线性应变矩阵已经做过对应交换，才是本程序计算出的结果。

        参考文献：

        [1] 连续体和结构的非线性有限元_Ted Belytschko等著 庄茁译_2002

        [2] 非线性有限元分析-张汝清-1990

        [3] Non-linear finite element analysis of solids and structures. 2012.

        """
        if self.dimension == 2:
            self.qp_bnl_matrices = np.zeros(shape=(self.qp_number, 4, self.element_dof_number), dtype=DTYPE)
        elif self.dimension == 3:
            self.qp_bnl_matrices = np.zeros(shape=(self.qp_number, 9, self.element_dof_number), dtype=DTYPE)

        if self.method == "TL":
            for iqp, (qp_shape_gradient, qp_jacobi_inv) in enumerate(zip(self.iso_element_shape.qp_shape_gradients, self.qp_jacobi_invs)):
                qp_dhdx = np.dot(qp_shape_gradient.transpose(), qp_jacobi_inv)
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
                qp_dhdx_t = np.dot(qp_shape_gradient.transpose(), qp_jacobi_inv_t)
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

        self.qp_bnl_matrices_transpose = np.array([qp_bnl_matrix.transpose() for qp_bnl_matrix in self.qp_bnl_matrices])

    def update_element_material_stiffness_fint(self,
                                               is_update_material: bool = True,
                                               is_update_stiffness: bool = True,
                                               is_update_fint: bool = True, ) -> None:
        element_id = self.element_id
        timer = self.timer
        ntens = self.ntens
        ndi = self.ndi
        nshr = self.nshr

        if is_update_stiffness:
            self.element_stiffness = np.zeros(shape=(self.element_dof_number, self.element_dof_number), dtype=DTYPE)

        if is_update_fint:
            self.element_fint = np.zeros(self.element_dof_number, dtype=DTYPE)

        if is_update_material:
            self.qp_ddsddes = list()
            self.qp_strains = list()
            self.qp_stresses = list()
            self.update_kinematics()
            self.create_qp_b_matrices()

        qp_number = self.qp_number
        qp_b_matrices = self.qp_b_matrices
        qp_b_matrices_transpose = self.qp_b_matrices_transpose
        qp_bnl_matrices = self.qp_bnl_matrices
        qp_bnl_matrices_transpose = self.qp_bnl_matrices_transpose
        qp_weight_times_jacobi_dets = self.qp_weight_times_jacobi_dets

        qp_state_variables = self.qp_state_variables
        qp_state_variables_new = self.qp_state_variables_new

        material_data = self.material_data_list[0]

        # 调换了上面代码的顺序，需要先执行self.update_kinematics()和self.create_qp_b_matrices()函数，否则变量作为副本未被更新。

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
                self.qp_stresses.append(qp_stress)
            else:
                qp_ddsdde = self.qp_ddsddes[i]
                qp_stress = self.qp_stresses[i]

            if is_update_stiffness:
                qp_stress_matrix = self.voigt_to_block_diagonal_matrix(qp_stress)
                if self.method == 'TL':
                    self.element_stiffness += np.dot(qp_b_matrix_transpose, np.dot(qp_ddsdde, qp_b_matrix)) * qp_weight_times_jacobi_det
                    self.element_stiffness += np.dot(qp_bnl_matrix_transpose, np.dot(qp_stress_matrix, qp_bnl_matrix)) * qp_weight_times_jacobi_det
                elif self.method == 'UL':
                    self.element_stiffness += np.dot(qp_b_matrix_transpose, np.dot(qp_ddsdde, qp_b_matrix)) * self.qp_weight_times_jacobi_dets_t[i]
                    self.element_stiffness += np.dot(qp_bnl_matrix_transpose, np.dot(qp_stress_matrix, qp_bnl_matrix)) * self.qp_weight_times_jacobi_dets_t[i]

            if is_update_fint:
                if self.method == 'TL':
                    self.element_fint += np.dot(qp_b_matrix_transpose, qp_stress) * qp_weight_times_jacobi_det
                elif self.method == 'UL':
                    self.element_fint += np.dot(qp_b_matrix_transpose, qp_stress) * self.qp_weight_times_jacobi_dets_t[i]

    def update_element_field_variables(self) -> None:
        self.qp_field_variables['strain'] = np.array(self.qp_strains, dtype=DTYPE) + np.array(self.qp_dstrains, dtype=DTYPE)
        self.qp_field_variables['stress'] = np.array(self.qp_stresses, dtype=DTYPE)
        for key in self.qp_state_variables_new[0].keys():
            if key not in ['strain', 'stress']:
                variable = []
                for qp_state_variable_new in self.qp_state_variables_new:
                    variable.append(qp_state_variable_new[key])
                self.qp_field_variables[f'SDV-{key}'] = np.array(variable, dtype=DTYPE)
        self.element_nodal_field_variables = set_element_field_variables(self.qp_field_variables, self.iso_element_shape, self.dimension)

    def voigt_to_block_diagonal_matrix(self, stress):
        T = np.zeros(shape=(self.dimension * self.dimension, self.dimension * self.dimension))

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

    job = Job(r'..\..\..\examples\test\plane\4element\Job-TL.toml')

    job.assembly.element_data_list[3].show()

    # e = SolidFiniteStrain(element_id=job.assembly.element_data_list[0].element_id,
    #                       iso_element_shape=job.assembly.element_data_list[0].iso_element_shape,
    #                       connectivity=job.assembly.element_data_list[0].connectivity,
    #                       node_coords=job.assembly.element_data_list[0].node_coords,
    #                       dof=job.assembly.element_data_list[0].dof,
    #                       materials=job.assembly.element_data_list[0].materials,
    #                       section=job.assembly.element_data_list[0].section,
    #                       material_data_list=job.assembly.element_data_list[0].material_data_list,
    #                       timer=job.assembly.element_data_list[0].timer)
    #
    # e.show()
