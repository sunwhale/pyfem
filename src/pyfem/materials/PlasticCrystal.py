# -*- coding: utf-8 -*-
"""

"""
from copy import deepcopy

import numpy as np
from numpy import zeros, ndarray, sign, dot, array, einsum, eye, ones, maximum, abs, transpose, all, delete, concatenate
from numpy.linalg import solve, inv

from pyfem.fem.Timer import Timer
from pyfem.fem.constants import DTYPE
from pyfem.io.Material import Material
from pyfem.io.Section import Section
from pyfem.materials.BaseMaterial import BaseMaterial
from pyfem.materials.crystal_slip_system import generate_mn
from pyfem.utils.colors import error_style
from pyfem.utils.mechanics import get_transformation, get_voigt_transformation


class PlasticCrystal(BaseMaterial):
    r"""
    晶体塑性材料。

    支持的截面属性：('Volume', 'PlaneStrain')

    :ivar tolerance: 误差容限
    :vartype tolerance: float

    :ivar total_number_of_slips: 总滑移系数量
    :vartype total_number_of_slips: int

    :ivar elastic: 弹性参数字典
    :vartype elastic: dict

    :ivar C: 弹性矩阵
    :vartype C: ndarray

    :ivar slip_system_name: 滑移系统名称
    :vartype slip_system_name: list[str]

    :ivar c_over_a: 晶体坐标系的c/a
    :vartype c_over_a: list[float]

    :ivar theta: 切线系数法参数
    :vartype theta: float

    :ivar K: 参考屈服强度
    :vartype K: float

    :ivar dot_gamma_0: 参考剪切应变率
    :vartype dot_gamma_0: float

    :ivar p_s: 强化指数
    :vartype p_s: float

    :ivar H: 硬化系数矩阵
    :vartype H: ndarray

    :ivar u_global: 全局坐标系下的1号矢量
    :vartype u_global: ndarray

    :ivar v_global: 全局坐标系下的2号矢量
    :vartype v_global: ndarray

    :ivar w_global: 全局坐标系下的3号矢量
    :vartype w_global: ndarray

    :ivar u_grain: 晶粒坐标系下的1号矢量
    :vartype u_grain: ndarray

    :ivar v_grain: 晶粒坐标系下的2号矢量
    :vartype v_grain: ndarray

    :ivar w_grain: 晶粒坐标系下的3号矢量
    :vartype w_grain: ndarray

    :ivar T: 坐标变换矩阵
    :vartype T: ndarray

    :ivar T_vogit: Vogit坐标变换矩阵
    :vartype T_vogit: ndarray

    :ivar m_s: 特征滑移系滑移方向
    :vartype m_s: ndarray

    :ivar n_s: 特征滑移系滑移面法向
    :vartype n_s: ndarray

    :ivar MAX_NITER: 最大迭代次数
    :vartype MAX_NITER: ndarray
    """

    __slots_dict__: dict = {
        'tolerance': ('float', '误差容限'),
        'total_number_of_slips': ('int', '总滑移系数量'),
        'elastic': ('dict', '弹性参数字典'),
        'C': ('ndarray', '弹性矩阵'),
        'slip_system_name': ('list[str]', '滑移系统名称'),
        'c_over_a': ('list[float]', '晶体坐标系的c/a'),
        'theta': ('float', '切线系数法参数'),
        'K': ('ndarray', '参考屈服强度'),
        'dot_gamma_0': ('ndarray', '参考剪切应变率'),
        'p_s': ('ndarray', '强化指数'),
        'c_1': ('ndarray', '随动强化参数'),
        'c_2': ('ndarray', '随动强化参数'),
        'r_0': ('ndarray', '初始阻应力'),
        'b': ('ndarray', '各项同性强化参数'),
        'Q': ('ndarray', '各项同性强化参数'),
        'H': ('ndarray', '硬化系数矩阵'),
        'u_global': ('ndarray', '全局坐标系下的1号矢量'),
        'v_global': ('ndarray', '全局坐标系下的2号矢量'),
        'w_global': ('ndarray', '全局坐标系下的3号矢量'),
        'u_grain': ('ndarray', '晶粒坐标系下的1号矢量'),
        'v_grain': ('ndarray', '晶粒坐标系下的2号矢量'),
        'w_grain': ('ndarray', '晶粒坐标系下的3号矢量'),
        'T': ('ndarray', '坐标变换矩阵'),
        'T_vogit': ('ndarray', 'Vogit坐标变换矩阵'),
        'm_s': ('ndarray', '特征滑移系滑移方向'),
        'n_s': ('ndarray', '特征滑移系滑移面法向'),
        'MAX_NITER': ('ndarray', '硬化系数矩阵'),
    }

    __slots__ = BaseMaterial.__slots__ + [slot for slot in __slots_dict__.keys()]

    def __init__(self, material: Material, dimension: int, section: Section) -> None:
        super().__init__(material, dimension, section)
        self.allowed_section_types = ('Volume', 'PlaneStrain')

        self.data_keys = []

        if len(self.material.data) != len(self.data_keys):
            raise NotImplementedError(error_style(self.get_data_length_error_msg()))
        else:
            for i, key in enumerate(self.data_keys):
                self.data_dict[key] = material.data[i]

        # 迭代及收敛参数
        self.tolerance: float = 1.0e-6
        self.MAX_NITER = 8
        self.theta: float = material.data_dict['theta']

        # 弹性参数
        self.elastic: dict = material.data_dict['elastic']
        self.C: ndarray = self.create_elastic_stiffness(self.elastic)

        # 滑移系参数
        self.total_number_of_slips: int = 0
        self.slip_system_name: list[str] = material.data_dict['slip_system_name']
        self.c_over_a: list[float] = material.data_dict['c_over_a']

        # 多滑移系赋值
        for i, (name, ca) in enumerate(zip(self.slip_system_name, self.c_over_a)):
            slip_system_number, m_s, n_s = generate_mn('slip', name, ca)
            self.total_number_of_slips += slip_system_number
            K = ones((slip_system_number,), dtype=DTYPE) * material.data_dict['K'][i]
            dot_gamma_0 = ones((slip_system_number,), dtype=DTYPE) * material.data_dict['dot_gamma_0'][i]
            p_s = ones((slip_system_number,), dtype=DTYPE) * material.data_dict['p_s'][i]
            c_1 = ones((slip_system_number,), dtype=DTYPE) * material.data_dict['c_1'][i]
            c_2 = ones((slip_system_number,), dtype=DTYPE) * material.data_dict['c_2'][i]
            r_0 = ones((slip_system_number,), dtype=DTYPE) * material.data_dict['r_0'][i]
            b = ones((slip_system_number,), dtype=DTYPE) * material.data_dict['b'][i]
            Q = ones((slip_system_number,), dtype=DTYPE) * material.data_dict['Q'][i]
            if i == 0:
                self.m_s: ndarray = m_s
                self.n_s: ndarray = n_s
                self.K: ndarray = K
                self.dot_gamma_0: ndarray = dot_gamma_0
                self.p_s: ndarray = p_s
                self.c_1: ndarray = c_1
                self.c_2: ndarray = c_2
                self.r_0: ndarray = r_0
                self.b: ndarray = b
                self.Q: ndarray = Q
            else:
                self.m_s = concatenate((self.m_s, m_s))
                self.n_s = concatenate((self.n_s, n_s))
                self.K = concatenate((self.K, K))
                self.dot_gamma_0 = concatenate((self.dot_gamma_0, dot_gamma_0))
                self.p_s = concatenate((self.p_s, p_s))
                self.c_1 = concatenate((self.c_1, c_1))
                self.c_2 = concatenate((self.c_2, c_2))
                self.r_0 = concatenate((self.r_0, r_0))
                self.b = concatenate((self.b, b))
                self.Q = concatenate((self.Q, Q))

        self.H = ones(shape=(self.total_number_of_slips, self.total_number_of_slips), dtype=DTYPE)

        # 晶粒取向信息
        self.u_global: ndarray = array(section.data_dict['u_global'])
        self.v_global: ndarray = array(section.data_dict['v_global'])
        self.w_global: ndarray = array(section.data_dict['w_global'])

        self.u_grain: ndarray = array(section.data_dict['u_grain'])
        self.v_grain: ndarray = array(section.data_dict['v_grain'])
        self.w_grain: ndarray = array(section.data_dict['w_grain'])

        self.T: ndarray = get_transformation(self.u_grain, self.v_grain, self.w_grain, self.u_global, self.v_global, self.w_global)
        self.T_vogit: ndarray = get_voigt_transformation(self.T)

        # 旋转至全局坐标系
        self.m_s = dot(self.m_s, self.T)
        self.n_s = dot(self.n_s, self.T)
        self.C = dot(dot(self.T_vogit, self.C), transpose(self.T_vogit))

    def create_elastic_stiffness(self, elastic: dict):
        r"""
        **定义局部晶系的弹性刚度矩阵**

        弹性刚度矩阵由弹性常数组成，对应的矩阵形式与弹性常数个数及材料对称性相关，相关参数由材料属性数据字典中的:py:attr:`elastic` 字典给出。

        （1）各向同性材料(Isotropic material)：对于一般各向同性材料，其包含两个独立的弹性常数，即：杨氏模量(Young's modulus) :math:`E` 和泊松比(Poisson's ratio) :math:`\nu` ,
        进一步可得到这两个弹性常数与剪切模量 :math:`G = \mu` 和拉梅常数 :math:`\lambda` 的关系为([1],[2])：

        .. math::
            \lambda  = \frac{{\nu E}}{{(1 + \nu )(1 - 2\nu )}},G = \mu  = \frac{E}{{2(1 + \nu )}}

        定义两个常数： :math:`E11 = \lambda  + 2\mu` 、 :math:`E12 = \lambda` ，得到弹性矩阵形式为：

        .. math::
            {{\mathbf{C}}_{(2D)}} = \left[ {\begin{array}{*{20}{c}}
              {E11}&{E12}&0 \\
              {E12}&{E11}&0 \\
              0&0&G
            \end{array}} \right]

        .. math::
            {{\mathbf{C}}_{(3D)}} = \left[ {\begin{array}{*{20}{c}}
              {E11}&{E12}&{E12}&0&0&0 \\
              {E12}&{E11}&{E12}&0&0&0 \\
              {E12}&{E12}&{E11}&0&0&0 \\
              0&0&0&G&0&0 \\
              0&0&0&0&G&0 \\
              0&0&0&0&0&G
            \end{array}} \right]

        [1] I.S. Sokolnikoff: Mathematical Theory of Elasticity. New York, 1956.

        [2] T.J.R. Hughes: The Finite Element Method, Linear Static and Dynamic Finite Element Analysis. New Jersey, 1987.

        （2）立方材料(Cubic material)：包含三个独立的材料参数 :math:`C11,C12,C44` ，其弹性矩阵定义为：

        .. math::
            {{\mathbf{C}}_{(2D)}} = \left[ {\begin{array}{*{20}{c}}
              {C11}&{C12}&0 \\
              {C12}&{C11}&0 \\
              0&0&{C44}
            \end{array}} \right]

        .. math::
            {{\mathbf{C}}_{(3D)}} = \left[ {\begin{array}{*{20}{c}}
              {C11}&{C12}&{C12}&0&0&0 \\
              {C12}&{C11}&{C12}&0&0&0 \\
              {C12}&{C12}&{C11}&0&0&0 \\
              0&0&0&{C44}&0&0 \\
              0&0&0&0&{C44}&0 \\
              0&0&0&0&0&{C44}
            \end{array}} \right]

        （3）正交材料(Orthotropic material)：包含9个独立的材料参数，分别为：

         .. math::
            D1111,D1122,D2222,D1133,D2233,D3333,D1212,D1313,D2323

        与 ABAQUS 对各向同性材料的定义相同，其弹性矩阵定义为：

        .. math::
            {{\mathbf{C}}_{(2D)}} = \left[ {\begin{array}{*{20}{c}}
              {D1111}&{D1122}&0 \\
              {D1122}&{D2222}&0 \\
              0&0&{D1212}
            \end{array}} \right]

        .. math::
            {{\mathbf{C}}_{(3D)}} = \left[ {\begin{array}{*{20}{c}}
              {D1111}&{D1122}&{D1133}&0&0&0 \\
              {D1122}&{D2222}&{D2233}&0&0&0 \\
              {D1133}&{D2233}&{D3333}&0&0&0 \\
              0&0&0&{D1212}&0&0 \\
              0&0&0&0&{D1313}&0 \\
              0&0&0&0&0&{D2323}
            \end{array}} \right]

        （4）各向异性材料(Anistropic material)：包含21个独立的材料参数，分别为：

        .. math::
            \begin{gathered}
              D1111,D1122,D2222,D1133,D2233,D3333,D1112, \hfill \\
              D2212,D3312,D1212,D1113,D2213,D3313,D1213, \hfill \\
              D1313,D1123,D2223,D3323,D1223,D1323,D2323 \hfill \\
            \end{gathered}

        与 ABAQUS 对各向异性材料的定义相同，其弹性矩阵定义为：

        .. math::
            {{\mathbf{C}}_{(2D)}} = \left[ {\begin{array}{*{20}{c}}
              {D1111}&{D1122}&0 \\
              {D1122}&{D2222}&0 \\
              0&0&{D1212}
            \end{array}} \right]

        .. math::
            {{\mathbf{C}}_{(3D)}} = \left[ {\begin{array}{*{20}{c}}
              {D1111}&{D1122}&{D1133}&{D1112}&{D1113}&{D1123} \\
              {D1122}&{D2222}&{D2233}&{D2212}&{D2213}&{D2223} \\
              {D1133}&{D2233}&{D3333}&{D3312}&{D3313}&{D3323} \\
              {D1112}&{D2212}&{D3312}&{D1212}&{D1213}&{D1223} \\
              {D1113}&{D2213}&{D3313}&{D1213}&{D1313}&{D1323} \\
              {D1123}&{D2223}&{D3323}&{D1223}&{D1323}&{D2323}
            \end{array}} \right]
        """
        symmetry = elastic['symmetry']
        if symmetry == 'isotropic':
            C11 = elastic['C11']
            C12 = elastic['C12']
            C44 = elastic['C44']
            C = array([[C11, C12, C12, 0, 0, 0],
                       [C12, C11, C12, 0, 0, 0],
                       [C12, C12, C11, 0, 0, 0],
                       [0, 0, 0, C44, 0, 0],
                       [0, 0, 0, 0, C44, 0],
                       [0, 0, 0, 0, 0, C44]], dtype=DTYPE)
        else:
            raise NotImplementedError(
                error_style(f'the symmetry type \"{symmetry}\" of elastic stiffness is not supported'))
        return C

    def get_tangent(self, variable: dict[str, ndarray],
                    state_variable: dict[str, ndarray],
                    state_variable_new: dict[str, ndarray],
                    element_id: int,
                    iqp: int,
                    ntens: int,
                    ndi: int,
                    nshr: int,
                    timer: Timer) -> tuple[ndarray, dict[str, ndarray]]:
        r"""
        **计算 ddsdde 矩阵与输出变量字典**

        本模块中包含3个字典：:py:attr:`variable` ， :py:attr:`state_variable` ， :py:attr:`state_variable_new` 。

        其中，字典 :py:attr:`variable` 存储已知的变量，如应变 :math:`\varepsilon` 和应变增量 :math:`\Delta \varepsilon` 。

        字典 :py:attr:`state_variable` 存储迭代过程中第 :math:`n` 个迭代步的状态变量，如应力 :math:`\sigma` 、分解剪应力 :math:`\tau` 、
        剪切应变 :math:`\gamma` 、状态变量 :math:`\rho` 、背应力项(随动强化项) :math:`\alpha` 、各向同性强化项 :math:`r` 、
        特征滑移系滑移方向 :math:`m\_s` 、特征滑移系滑移面法向 :math:`n\_s` 。这些状态变量在计算收敛之前，不断被更新。

        字典 :py:attr:`state_variable_new` 存储迭代收敛时的状态变量。

        本文中的滑移系剪切应变演化唯象模型建立包含以下几个部分：

        （1）弹塑性本构：

        晶体在变形过程中会旋转，但只有材料的拉伸才是应变，旋转不算应变。所以普通的增量形式本构方程 :math:`\hat{\boldsymbol{\sigma}}= \mathbb{C}:{{\boldsymbol{D}}}` 不再适用，
        需要做出相应的改变。主要改变有两点：

        1. 用焦曼应力率 :math:`\hat{\boldsymbol{\sigma}}^{\mathrm{e}}` 取代普通的应力率，消除旋转带来的影响，保证客观性。 :math:`\mathbb{C}` 为弹性模量张量。
        (焦曼应力率是一种客观率(objective rate)，客观率是对应力变化率的一个测定，它使得在刚体转动中，在初始参考系下，初始的应力状态保持不变，即在刚体旋转的情况下为 0，
        这样的 stress rate 就叫做 objective rate，也叫做 frame-invariant rate[1])。

        2. 选用一个客观的应变率。本文中选用弹性变形率张量 :math:`{{\boldsymbol{D}}^{\rm{e}}}` ， :math:`{{\boldsymbol{D}}^{\rm{e}}}` 的客观性是因为其中不包含旋转的影响。

        根据以上两点，我们得到消除旋转影响的弹性本构关系，可将其写为：

        .. math::
            \hat{\boldsymbol{\sigma}}^{\mathrm{e}} = \mathbb{C}:{{\boldsymbol{D}}^{\rm{e}}}

        要得到上式的弹性本构关系，首先需要根据晶体塑性变形几何学与运动学得到 :math:`{{\boldsymbol{D}}^{\rm{e}}}` 。

        （2）晶体塑性变形几何学与运动学公式推导：

        由于晶内部大量位错的存在，所以宏观上可以假设位错滑移在晶粒内部均匀分布。因而，在连续介质力学中，用变形梯度张量 :math:`F` 来描述滑移变形的宏观效应。 :math:`F`
        里包含了变形的所有信息，包括弹性变形和塑性变形，也包括了拉伸和旋转。采用 Hill 和 Rice 对晶体塑性变形几何学及运动学的理论表述方法，
        则晶体总的变形梯度 :math:`F` ——当前构型量 :math:`x` 对参考构型量 :math:`X` 的偏导，可表示为：

        .. math::
            {\boldsymbol{F}} =\frac{\partial {x}}{\partial {X}} = {{\boldsymbol{F}}^{\text{e}}}{{\boldsymbol{F}}^{\text{p}}}

        其中， :math:`{{\boldsymbol{F}}^{\text{e}}}` 为晶格畸变和刚性转动产生的弹性变形梯度， :math:`{{\boldsymbol{F}}^{\text{p}}}`
        表示晶体沿着滑移方向的均匀剪切所产生的塑性变形梯度。

        晶体变形几何学示意图::

                                                                    *
                                                               *      *
                                                           *    *
                                                     *      *    *
                                              *       *      *
                                               *       *      *
                                         *      *       *     m^*      |\
                                          *      *       *               \
                                    *      *      *                       \
                            n^*      *      *                              \
                              *       *                                     \  F^e
                               *                                             \
                      /|                                                      \
                    /                                                          \
                  /  F=F^eF^p                                                   \
            n   /                                                                 *  *  *  *  *
            *  *  *  *  *                   F^p                              *  *  *  *  *  *
            *  *  *  *  *           ---------------------->              n  *  *  *  *  *  *
            *  *  *  *  *                                                *  *  *  *  *  *
            *  *  *  *  * m                                              *  *  *  *  * m


        上图所示为晶体变形几何学的示意图。可以看出，晶体滑移过程中，晶格矢量没有发生变化；但晶格的畸变会造成晶格矢量的变化，包括伸长和转动。

        用 :math:`{\boldsymbol{m}}^{(\alpha )}` 和 :math:`{\boldsymbol{n}}^{(\alpha )}` 分别表示变形前第 :math:`\alpha` 滑移系滑移方向和滑移面法向的单位向量。
        用 :math:`{\boldsymbol{m}}^{*\left( \alpha  \right)}` 和 :math:`{\boldsymbol{m}}^{*\left( \alpha  \right)}` 分别表示晶格畸变后第 :math:`\alpha`
        滑移系的滑移方向和滑移面法向的单位向量。变形前与变形后第 :math:`\alpha` 滑移系的滑移方向和滑移面法向的单位向量存在下列关系：

        .. math::
            {{\boldsymbol{m}}^{*\left( \alpha  \right)}} = {{\boldsymbol{F}}^{\text{e}}}{{\boldsymbol{m}}^{\left( \alpha  \right)}},
            {{\boldsymbol{n}}^{*\left( \alpha  \right)}} = {{\boldsymbol{n}}^{\left( \alpha  \right)}} {\left( {{{\boldsymbol{F}}^{\rm{e}}}} \right)^{{\rm{ - }}1}}

        晶格畸变后，滑移面的滑移方向 :math:`{\boldsymbol{m}}^{*\left( \alpha  \right)}` 和法线方向 :math:`{\boldsymbol{m}}^{*\left( \alpha  \right)}` 不再是单位向量，
        但仍保持正交。

        自然的，可定义变形速度梯度，即变形速度 :math:`v` 对当前构型 :math:`x` 的导数，也被称为速度梯度张量 :math:`{\boldsymbol{L}}` ：

        .. math::
            {\boldsymbol{L}} = \frac{{\partial {{v}}}}{{\partial {{x}}}} = {\boldsymbol{\dot F}}{{\boldsymbol{F}}^{ - 1}}

        这个速度梯度张量 :math:`L` 也包含了旋转的影响，因此我们需要对其进行分解，得到不含旋转的弹性变形率张量  :math:`{{\boldsymbol{D}}^{\rm{e}}}` 。该分解分为两步：

        第一步，除去塑性变形的部分：

        对应于前述变形梯度的乘法分解，将速度梯度分解为与晶格畸变和刚体转动相对应的弹性部分 :math:`{{\boldsymbol{L}}^{\rm{e}}}`
        和与滑移相对应的塑性部分 :math:`{{\boldsymbol{L}}^{\rm{p}}}` ：

        .. math::
            {\boldsymbol{L}} = {{\boldsymbol{L}}^{\text{e}}}{{\boldsymbol{L}}^{\text{p}}}

        其中， :math:`{{\boldsymbol{L}}^{\rm{e}}}` 和 :math:`{{\boldsymbol{L}}^{\rm{p}}}` 分别为：

        .. math::
           {{\boldsymbol{L}}^{\rm{e}}} = {{{\boldsymbol{\dot F}}}^{\rm{e}}}{\left( {{{\boldsymbol{F}}^{\rm{e}}}} \right)^{ - 1}},
           \boldsymbol{L}^{\mathrm{p}}=\boldsymbol{F}^{\mathrm{e}} \dot{\boldsymbol{F}}^{\mathrm{p}}
           \left(\boldsymbol{F}^{\mathrm{p}}\right)^{-1}\left(\boldsymbol{F}^{\mathrm{e}}\right)^{-1}

        假设晶体整体的塑性变形由各滑移系中滑移引起的剪切应变所确定，在初始构型或中间构型上，有：

        .. math::
            {{\boldsymbol{L}}^{\rm{p}}} =  \dot{\boldsymbol{F}}^{\mathrm{p}}\left(\boldsymbol{F}^{\mathrm{p}}\right)^{-1}
            = \sum\limits_{\alpha  = 1}^N {{{\dot \gamma }^{\left( \alpha  \right)}}}
            {{\boldsymbol{m}}^{\left( \alpha  \right)}} \otimes {{\boldsymbol{n}}^{\left( \alpha  \right)}}

        其中， :math:`{{{\dot \gamma }^{\left( \alpha  \right)}}}` 表示第 :math:`\alpha` 个滑移系的滑移剪切率，求和将对所有激活的滑移系进行。

        利用变形前与变形后第 :math:`\alpha` 滑移系的滑移方向和滑移面法向单位向量的关系，得到最终构型的 :math:`{{\boldsymbol{L}}^{\rm{p}}}` 表达式为：

        .. math::
            {{\boldsymbol{L}}^{\rm{p}}} = \sum\limits_{\alpha  = 1}^N {{{\dot \gamma }^{\left( \alpha  \right)}}}
            {{\boldsymbol{F}}^{\rm{e}}}{{\boldsymbol{m}}^{\left( \alpha  \right)}} \otimes {{\boldsymbol{n}}^{\left( \alpha  \right)}}{\left(
            {{{\boldsymbol{F}}^{\rm{e}}}} \right)^{ - 1}} = \sum\limits_{\alpha  = 1}^N {{{\dot \gamma }^{\left( \alpha  \right)}}}
            {{\boldsymbol{m}}^{*\left( \alpha  \right)}} \otimes {{\boldsymbol{n}}^{*\left( \alpha  \right)}}

        第二步，除去旋转的部分：

        速度梯度可以进一步分解为对称部分和反对称部分之和：

        .. math::
            {\boldsymbol{L}} = \frac{1}{2}\left( {{\boldsymbol{L}} + {{\boldsymbol{L}}^{\rm{T}}}} \right) + \frac{1}{2}\left( {{\boldsymbol{L}} - {{\boldsymbol{L}}^{\rm{T}}}} \right)

        将速度梯度的对称部分定义为变形率张量：

        .. math::
            {\boldsymbol{D}} = \frac{1}{2}\left( {{\boldsymbol{L}} + {{\boldsymbol{L}}^{\rm{T}}}} \right)={{\boldsymbol{D}}^{\rm{e}}}+{{\boldsymbol{D}}^{\rm{p}}}

        将速度梯度的反对称部分定义为旋率张量：

        .. math::
            {\boldsymbol{W}} = \frac{1}{2}\left( {{\boldsymbol{L}} - {{\boldsymbol{L}}^{\rm{T}}}} \right)={{\boldsymbol{W}}^{\rm{e}}}{\rm{ + }}{{\boldsymbol{W}}^{\rm{p}}}

        将变形率张量 :math:`D` 分解得到弹性变形率张量 :math:`{{\boldsymbol{D}}^{\rm{e}}}` 与塑性变形率张量 :math:`{{\boldsymbol{D}}^{\rm{p}}}` ：

        .. math::
            {{\boldsymbol{D}}^{\rm{e}}}= \frac{1}{2}\left( {{{\boldsymbol{L}}^{\rm{e}}} + {{\left( {{{\boldsymbol{L}}^{\rm{e}}}}
            \right)}^{\rm{T}}}} \right) = \frac{1}{2}\left( {{{{\boldsymbol{\dot F}}}^{\rm{e}}}{{\left( {{{\boldsymbol{F}}^{\rm{e}}}}
            \right)}^{ - 1}} + {{\left( {{{{\boldsymbol{\dot F}}}^{\rm{e}}}{{\left( {{{\boldsymbol{F}}^{\rm{e}}}} \right)}^{ - 1}}}
            \right)}^{\rm{T}}}} \right)

        .. math::
            {{\boldsymbol{D}}^{\rm{p}}}= \frac{1}{2}\left( {{{\boldsymbol{L}}^{\rm{p}}} + {{\left( {{{\boldsymbol{L}}^{\rm{p}}}}
            \right)}^{\rm{T}}}} \right) = \frac{1}{2}\left( {\sum\limits_{\alpha  = 1}^N {{{\dot \gamma }^{\left(
            \alpha  \right)}}} {{\boldsymbol{m}}^{*\left( \alpha  \right)}} \otimes {{\boldsymbol{n}}^{*\left( \alpha  \right)}} +
            {{\left( {\sum\limits_{\alpha  = 1}^N {{{\dot \gamma }^{\left( \alpha  \right)}}} {{\boldsymbol{m}}^{*\left( \alpha
            \right)}} \otimes {{\boldsymbol{n}}^{*\left( \alpha  \right)}}} \right)}^{\rm{T}}}} \right)

        .. math::
            {{\boldsymbol{D}}^{\rm{p}}} = \sum\limits_{\alpha  = 1}^N {\frac{1}{2}\left( {{{\boldsymbol{m}}^{*\left( \alpha  \right)}}
            \otimes {{\boldsymbol{n}}^{*\left( \alpha  \right)}} + {{\boldsymbol{n}}^{*\left( \alpha  \right)}} \otimes {{\boldsymbol{m}}^{*
            \left(\alpha  \right)}}} \right){{\dot \gamma }^{\left( \alpha  \right)}}}  = \sum\limits_{\alpha  = 1}^N
            {{{\boldsymbol{P}}^{\left( \alpha  \right)}}{{\dot \gamma }^{\left( \alpha  \right)}}}

        其中， :math:`{\boldsymbol{P}}^{\left( \alpha  \right)}` 类似施密特因子， :math:`{\boldsymbol{P}}^{\left( \alpha  \right)}` 的表达式为：

        .. math::
            {{\boldsymbol{P}}^{\left( \alpha  \right)}} = \frac{1}{2}\left( {{{\boldsymbol{m}}^{*\left( \alpha  \right)}} \otimes
            {{\boldsymbol{n}}^{*\left( \alpha  \right)}} + {{\boldsymbol{n}}^{*\left( \alpha  \right)}} \otimes {{\boldsymbol{m}}^{*\left(
            \alpha  \right)}}} \right)

        同理，可以得到弹性旋率张量 :math:`{{\boldsymbol{W}}^{\rm{e}}}` 和塑性旋率张量 :math:`{{\boldsymbol{W}}^{\rm{p}}}` ：

        .. math::
            {{\boldsymbol{W}}^{\rm{e}}} =  \frac{1}{2}\left( {{{\boldsymbol{L}}^{\rm{e}}} - {{\left( {{{\boldsymbol{L}}^{\rm{e}}}}
            \right)}^{\rm{T}}}} \right) = \frac{1}{2}\left( {{{{\boldsymbol{\dot F}}}^{\rm{e}}}{{\left( {{{\boldsymbol{F}}^{\rm{e}}}}
            \right)}^{ - 1}} - {{\left( {{{{\boldsymbol{\dot F}}}^{\rm{e}}}{{\left( {{{\boldsymbol{F}}^{\rm{e}}}} \right)}^{ - 1}}}
            \right)}^{\rm{T}}}} \right)

        .. math::
            {{\boldsymbol{W}}^{\rm{p}}} =  \frac{1}{2}\left( {{{\boldsymbol{L}}^{\rm{p}}} - {{\left( {{{\boldsymbol{L}}^{\rm{p}}}}
            \right)}^{\rm{T}}}} \right) = \sum\limits_{\alpha  = 1}^N {\frac{1}{2}\left( {{{\boldsymbol{m}}^{*\left( \alpha  \right)}}
            \otimes {{\boldsymbol{n}}^{*\left( \alpha  \right)}} - {{\boldsymbol{n}}^{*\left( \alpha  \right)}} \otimes {{\boldsymbol{m}}^{*
            \left( \alpha  \right)}}} \right){{\dot \gamma }^{\left( \alpha  \right)}}}  = \sum\limits_{\alpha  = 1}^N
            {{{\boldsymbol{\Omega}}^{\left( \alpha  \right)}}{{\dot \gamma }^{\left( \alpha  \right)}}}

        其中 :math:`\boldsymbol{\Omega}^{(\alpha)}` 的表达式为：

        .. math::
            {{\boldsymbol{\Omega }}^{\left( \alpha  \right)}}= \frac{1}{2}\left( {{{\boldsymbol{m}}^{*\left( \alpha  \right)}}
            \otimes {{\boldsymbol{n}}^{*\left( \alpha  \right)}}{\rm{ - }}{{\boldsymbol{n}}^{*\left( \alpha  \right)}} \otimes
            {{\boldsymbol{m}}^{*\left( \alpha  \right)}}} \right)

        以上晶体变形动力学的基本方程建立起了各滑移系剪切应变率与晶体宏观变形率之间的关系。

        **下面结合第一部分对弹塑性本构模型的介绍，将应力率、变形率及滑移剪切应变率联系起来：**

        将弹性本构表示为以中间构形为基准状态的 Kirchhoff应力张量 的 Jaumann导数。则有：

        .. math::
            \hat{\boldsymbol{\sigma}}^{\mathrm{e}} = {\boldsymbol {\dot \sigma }}  - {{\boldsymbol{W}}^{\rm{e}}} \cdot
            {\boldsymbol{\sigma}} + {\boldsymbol{\sigma}} \cdot {{\boldsymbol{W}}^{\rm{e}}}

        以初始构形为基础的柯西应力张量的 Zaremba-Jaumann率，简称 Jaumann率 表达式为：

        .. math::
            {\boldsymbol {\hat \sigma }} = {\boldsymbol {\dot \sigma }} - {\boldsymbol{W}} \cdot {\boldsymbol{\sigma}} +
            {\boldsymbol{\sigma}} \cdot {\boldsymbol{W}}

        结合上述两式，将 :math:`{\boldsymbol {\dot \sigma }}` 作替换， :math:`\hat{\boldsymbol{\sigma}}` 可以表示为：

        .. math::
            \hat{\boldsymbol{\sigma}}  =\hat{\boldsymbol{\sigma}}^{\mathrm{e}}-\boldsymbol{W}^{\mathrm{p}} \cdot
            \boldsymbol{\sigma}+\boldsymbol{\sigma} \cdot \boldsymbol{W}^{\mathrm{p}}  ={\mathbb{C}}:{{\boldsymbol{D}}^{\rm{e}}}
            -{{\boldsymbol{W}}^{\rm{p}}} \cdot {\boldsymbol{\sigma }} + {\boldsymbol{\sigma }} \cdot {{\boldsymbol{W}}^{\rm{p}}}

        将 :math:`{{\boldsymbol{D}}^{\rm{e}}}` 和 :math:`{{\boldsymbol{W}}^{\rm{p}}}` 作替换，得到：

        .. math::
            {\boldsymbol{\hat \sigma }}  = \mathbb{C}:\left( {{\boldsymbol{D}} - {{\boldsymbol{D}}^{\rm{p}}}} \right) -
            \sum\limits_{\alpha  = 1}^N {{{\boldsymbol{B}}^{\left( \alpha  \right)}}{{\dot \gamma }^{\left( \alpha  \right)}}}

        其中，

        .. math::
            {{\boldsymbol{B}}^{\left( \alpha  \right)}} = {{\boldsymbol {\Omega }}^{\left( \alpha  \right)}} \cdot {\boldsymbol{\sigma}}
            - {\boldsymbol{\sigma}} \cdot {{\rm{\Omega }}^{\left( \alpha  \right)}}

        得到：

        .. math::
            {\boldsymbol{\hat \sigma }}= \mathbb{C}:\left( {{\boldsymbol{D}} - \sum\limits_{\alpha  = 1}^N {{{\boldsymbol{P}}^{\left(
            \alpha  \right)}}{{\dot \gamma }^{\left( \alpha  \right)}}} } \right) - \sum\limits_{\alpha  = 1}^N
            {{{\boldsymbol{B}}^{\left( \alpha  \right)}}{{\dot \gamma }^{\left( \alpha  \right)}}}

        定义 :math:`\boldsymbol {S}^{(\alpha)}` 为：

        .. math::
            \boldsymbol {S}^{(\alpha)} = \mathbb{C}:{{\boldsymbol{P}}^{\left( \alpha  \right)}} +  {{\boldsymbol {\Omega }}^{\left( \alpha
            \right)}} \cdot {\boldsymbol{\sigma}} - {\boldsymbol{\sigma}} \cdot {{\rm{\Omega }}^{\left( \alpha  \right)}}

        最后得到 Jaumann率 表达式为：

        .. math::
            {\boldsymbol{\hat \sigma }}= \mathbb{C}:{\boldsymbol{D}} - \sum\limits_{\alpha  = 1}^N {\left[ {\mathbb{C}:
            {{\boldsymbol{P}}^{\left( \alpha  \right)}} +  {{\boldsymbol{\Omega }}^{\left( \alpha  \right)}} \cdot {\boldsymbol {\sigma }}
            - {\boldsymbol {\sigma }} \cdot {{\boldsymbol{\Omega }}^{\left( \alpha  \right)}}} \right]{{\dot \gamma }^{\left(
            \alpha  \right)}}}  = \mathbb{C}:{\boldsymbol{D}} - \sum\limits_{\alpha  = 1}^N { \boldsymbol {S}^{(\alpha)} {{\dot
            \gamma }^{\left( \alpha  \right)}}}

        上式将应力率、变形率及滑移剪切应变率联系起来，表示了应力率、变形率和滑移剪切应变率之间的定量关联。

        下一步的核心为确定所有可能开动滑移系的滑移剪切应变率 :math:`\dot{\gamma}^{(\alpha)}` 。

        （3）建立滑移系剪切应变演化唯象模型

        3.1 基础方程

        在晶体塑性本构模型中，需要通过各滑移系的剪切应变率计算应力率。因此，首先需要确定各滑移系剪切应变的演化方程。在剪切应变的硬化方程中，广泛地采用幂函数的形式，
        并且为了考虑晶体的循环塑性变形，引入各向同性强化项和随动强化项，建立如下混合强化模型：

        .. math::
            {{\dot \gamma }^{\left( \alpha  \right)}} =  {{\dot a}^{\left( \alpha  \right)}}{f^{\left( \alpha  \right)}}
            \left( {\frac{{\left| {{\tau ^{\left( \alpha  \right)}} - {\alpha ^{\left( \alpha  \right)}}} \right| -
            {r^{\left( \alpha  \right)}}}}{{{g^{\left( \alpha  \right)}}}}} \right) = {{\dot a}^{\left( \alpha
            \right)}}{\rm{sign}}\left( {{\tau ^{\left( \alpha  \right)}} - {\alpha ^{\left( \alpha  \right)}}}
            \right){\left\langle {\frac{{\left| {{\tau ^{\left( \alpha  \right)}} - {\alpha ^{\left( \alpha  \right)}}}
            \right| - {r^{\left( \alpha  \right)}}}}{{{g^{\left( \alpha  \right)}}}}} \right\rangle ^n}

        式中， :math:`\dot{\gamma}^{(\alpha)}` 为滑移系 :math:`\alpha` 的剪切应变率； :math:`\tau^{(\alpha)}` 为滑移系 :math:`\alpha`
        的分解剪应力； :math:`\alpha^{(\alpha)}` 和 :math:`r^{(\alpha)}` 分别为滑移系 :math:`\alpha` 的背应力项（随动强化项）和各向同性强化项。
        另外， :math:`\dot{a}^{(\alpha)}` 为滑移系 :math:`\alpha` 的参考剪切应变率， :math:`n` 为应变速率敏感指数，
        当  :math:`n \rightarrow \infty` 时，接近于应变速率无关的情况，但此时计算不稳定。 :math:`g^{(\alpha)}` 为参考屈服强度，是取决于温度和滑移系种类的材料常数。

        式中， :math:`\tau^{(\alpha)}` ， :math:`\alpha^{(\alpha)}` ， :math:`r^{(\alpha)}` 的增量形式可写为：

        .. math::
            \Delta \tau^{(\alpha)}=\dot{\tau}^{(\alpha)} \Delta t，
            \Delta \alpha^{(\alpha)}=\dot{\alpha}^{(\alpha)} \Delta t，
            \Delta r^{(\alpha)}=\dot{r}^{(\alpha)} \Delta t

        首先确定 :math:`\Delta \tau^{(\alpha)}` ，滑移系上的分解剪应力 :math:`{\tau ^{\left( \alpha  \right)}}` 定义为：

        .. math::
            {\tau ^{\left( \alpha  \right)}} = {{\boldsymbol{P}}^{\left( \alpha  \right)}}:{\rm{\sigma }}

        所以分解剪应力 :math:`{\tau ^{\left( \alpha  \right)}}` 对时间的导数 :math:`{\dot \tau }` 的表达形式为：

        .. math::
            {{\dot \tau }^{\left( \alpha  \right)}} = {{{\boldsymbol{\dot P}}}^{\left( \alpha  \right)}}:{\rm{\sigma }} +
            {{\boldsymbol{P}}^{\left( \alpha  \right)}}:{\rm{\dot \sigma }}

        其中， :math:`{{{\boldsymbol{\dot P}}}^{\left( \alpha  \right)}}` 可用下列式子表示:

        .. math::
            {{{\boldsymbol{\dot P}}}^{\left( \alpha  \right)}} = {{\boldsymbol{D}}^{\rm{e}}}{{\boldsymbol{\Omega }}^{\left( \alpha  \right)}}
            + {{\boldsymbol{W}}^{\rm{e}}}{{\boldsymbol{P}}^{\left( \alpha  \right)}} - {{\boldsymbol{\Omega }}^{\left( \alpha  \right)}}{{\boldsymbol{D}}^{\rm{e}}}
            - {{\boldsymbol{P}}^{\left( \alpha  \right)}}{{\boldsymbol{W}}^{\rm{e}}}

        将 :math:`{{{\boldsymbol{\dot P}}}^{\left( \alpha  \right)}}` 和 :math:`{\rm{\dot \sigma }}` 代入 :math:`{\dot \tau }`
        的表达式，并化简得到：

        .. math::
            {{\dot \tau }^{\left( \alpha  \right)}} =\left( {\mathbb{C}:{{\boldsymbol{P}}^{\left( \alpha
            \right)}} + {{\boldsymbol{\Omega }}^{\left( \alpha  \right)}}{\boldsymbol{\sigma }} - {\boldsymbol{\sigma }}
            {{\boldsymbol{\Omega }}^{\left( \alpha  \right)}}} \right):(\boldsymbol {D}-{{\boldsymbol{D}}^{\rm{p}}})

        将 :math:`\boldsymbol {S}^{(\alpha)}` 和 :math:`{{\boldsymbol{D}}^{\rm{p}}}` 代入上式得到：

        .. math::
            {{\dot \tau }^{\left( \alpha  \right)}}  = \boldsymbol {S}^{(\alpha)} :(\boldsymbol {D}-{{\boldsymbol{D}}^{\rm{p}}}) =
            \boldsymbol {S}^{(\alpha)} : \left( {\boldsymbol{D}}  - \sum\limits_{\alpha  = 1}^N {{{\boldsymbol{P}}^{\left( \alpha
            \right)}}{{\dot \gamma }^{\left( \alpha  \right)}}} \right)

        对 :math:`{\dot \tau }` 进行积分，得到分解剪应力离散格式：

        .. math::
            \Delta \tau^{(\alpha)} = {{\dot \tau }^{\left( \alpha  \right)}} \Delta t=\boldsymbol{S}^{(\alpha)}: \Delta
            \boldsymbol{\varepsilon}-\boldsymbol {S}^{(\alpha)}: \sum_{\beta=1}^{N} \boldsymbol {P}^{(\beta)} \Delta \gamma^{(\beta)}
            =  \boldsymbol{S}^{(\alpha)}: \left(\Delta \boldsymbol{\varepsilon}- \sum_{\beta=1}^{N} \boldsymbol {P}^{(\beta)} \Delta
            \gamma^{(\beta)} \right)

        然后确定 :math:`\Delta \alpha^{(\alpha)}` ，其中背应力 :math:`\dot{\alpha}^{(\alpha)}` 的演化方程为：

        .. math::
            \dot{\alpha}^{(\alpha)}=c_{1} \dot{\gamma}^{(\alpha)}-c_{2}\left|\dot{\gamma}^{(\alpha)}\right| \alpha^{(\alpha)}

        式中， :math:`c_{1} \dot{\gamma}^{(\alpha)}` 为背应力 :math:`\dot{\alpha}^{(\alpha)}` 关于剪切应变 :math:`\dot{\gamma}^{(\alpha)}`
        的线性项； :math:`c_{2}\left|\dot{\gamma}^{(\alpha)}\right| \alpha^{(\alpha)}` 为背应力 :math:`\dot{\alpha}^{(\alpha)}`
        关于剪切应变 :math:`\dot{\gamma}^{(\alpha)}` 的非线性项。 :math:`c_{1}` 和 :math:`c_{2}` 为取决于温度和滑移系种类的材料常数。

        得到背应力的离散格式为：

        .. math::
            \Delta \alpha^{(\alpha)}=c_{1} \Delta \gamma^{(\alpha)}-c_{2}\left|\Delta \gamma^{(\alpha)}\right| \alpha^{(\alpha)}

        最后确定 :math:`\Delta r^{(\alpha)}` ，其中各向同性强化项 :math:`\dot{r}^{(\alpha)}` 的演化方程为：

        .. math::
            {{\dot r}^{\left( \alpha  \right)}} = bQ\sum\limits_\beta  {{h_{\alpha \beta }}{{\dot \rho }^{\left( \beta  \right)}}}

        式中，引入各向同性状态变量  :math:`\rho^{(\beta)}` 描述晶体滑移中的位错硬化；引入交互作用系数矩阵  :math:`h_{\alpha \beta}` 描述滑移系间的交叉硬化，
        其对角线项表示滑移系的 “自硬化”，非对角线项表示由于滑移系间的耦合效应造成的 “潜在硬化”。对于给定的一组包含每个滑移系  :math:`\rho^{(\beta)}` 的位错状态，
        硬化将由系数 :math:`Q` 来确定。  :math:`b` 和  :math:`Q` 为取决于温度和滑移系种类的材料常数。因此有：

        .. math::
            r^{(\alpha)}=r_{0}^{(\alpha)}+b Q \sum_{\beta} h_{\alpha \beta} \rho^{(\beta)}

        滑移系 :math:`\alpha` 的各向同性强化项 :math:`r^{(\alpha)}` 由滑移系 :math:`\alpha` 的临界分解剪应力(初始阻应力) :math:`r_{0}^{(\alpha)}`
        和所有激活滑移系的位错状态变量 :math:`\rho^{(\beta)}` 共同确定。从冶金学的角度来看，
        该方程从左到右描述了基体带来的固溶硬化 :math:`r_{0}^{(\alpha)}` 和位错硬化 :math:`b Q \sum_{\beta} h_{\alpha \beta} \rho^{(\beta)}` 。

        在各向同性强化项 :math:`\dot{r}^{(\alpha)}` 中，采用非线性饱和形式的方程表示各向同性状态变量 :math:`\rho^{(\beta)}`，则有：

        .. math::
            \dot{\rho}^{(\beta)}=\left(1-b \rho^{(\beta)}\right)\left|\dot{\gamma}^{(\beta)}\right|

        其中，状态变量 :math:`\rho^{(\beta)}` 对应于可以进入 :math:`\gamma` 通道的临界位错密度。

        因此可以得到 :math:`\dot{r}^{(\alpha)}` 和 :math:`r^{(\alpha)}` 的完整表达式：

        .. math::
            \dot{r}^{(\alpha)}=b Q \sum_{\beta} h_{\alpha \beta}\left(1-b \rho^{(\beta)}\right)\left|\dot{\gamma}^{(\beta)}\right|

        .. math::
            r^{(\alpha)}=r_{0}^{(\alpha)}+b Q \sum_{\beta} h_{\alpha \beta}\left(1-b \rho^{(\beta)}\right)\left|\gamma^{(\beta)}\right|

        各向同性强化项离散格式为：

        .. math::
            \Delta r^{(\alpha)}=b Q \sum_{\beta} h_{\alpha \beta}\left(1-b \rho^{(\beta)}\right)\left|\Delta \gamma^{(\beta)}\right|

        通过上述推导，方程 :math:`{{\dot \gamma }^{\left( \alpha  \right)}}` 建立了能够描述晶体循环变形中各向同性强化和随动强化的剪切应变硬化方程。
        利用计算得到的各滑移系中的剪切应变增量和晶体塑性理论中的本构关系，即可得到宏观应力增量，下面将详细介绍混合强化模型的数值离散过程。

        3.2 数值离散求解

        如果将晶体塑性本构模型与基于位移场的求解有限元软件相结合，主要包含两个基本任务：一是通过积分点处的变形计算应力值，二是更新当前积分点的切线刚度矩阵。
        而应力和切线刚度矩阵的更新则依赖于所有开动滑移系的剪切应变增量 :math:`\Delta \gamma^{(\alpha)}` 的求解。

        3.2.1 求解应力应变增量以及相关内变量初值

        在速率相关晶体塑性本构模型的计算中，为了保证数值计算的稳定性，Peirce等人采用切线系数法改进方程 :math:`{{\dot \gamma }^{\left( \alpha  \right)}}` 的计算。
        令时间步长 :math:`\Delta t` 对应的剪切应变增量为 :math:`\Delta \gamma^{(\alpha)}` ，采用线性中心差分格式，则有：

        .. math::
            \Delta \gamma^{(\alpha)}=\Delta t\left[(1-\theta) \dot{\gamma}^{(\alpha)}(t)+\theta \dot{\gamma}^{(\alpha)}(t+\Delta t)\right]

        其中， :math:`\theta` 的取值范围为 0 至 1  :math:`(0 \leqslant \theta \leqslant 1)` ，其推荐取值范围为 0.5。
        用式 :math:`{{\dot \gamma }^{\left( \alpha  \right)}}` 的泰勒展开式近似地表示 :math:`\Delta \gamma^{(\alpha)}` 中的最后一项：

        .. math::
            {{\dot \gamma }^{\left( \alpha  \right)}}\left( {t + \Delta t} \right) = {{\dot \gamma }^{\left( \alpha
            \right)}}\left( t \right) + {\left. {\frac{{\partial {{\dot \gamma }^{(\alpha )}}}}{{\partial {\tau ^{(\alpha )}}}}}
            \right|_t}\Delta {\tau ^{\left( \alpha  \right)}} +{\left. {\frac{{\partial {{\dot \gamma }^{(\alpha )}}}}{{\partial
            {\alpha ^{(\alpha )}}}}} \right|_t}\Delta {\alpha ^{\left( \alpha  \right)}} + {\left. {\frac{{\partial {{\dot
            \gamma }^{(\alpha )}}}}{{\partial {r^{(\alpha )}}}}} \right|_t}\Delta {r^{\left( \alpha  \right)}}

        可得剪切应变增量方程 :math:`\Delta \gamma^{(\alpha)}` ：

        .. math::
            \Delta {\gamma ^{\left( \alpha  \right)}} = \Delta t\left[ {{{\dot \gamma }^{\left( \alpha  \right)}}\left(
            t \right) + \theta {{\left. {\frac{{\partial {{\dot \gamma }^{(\alpha )}}}}{{\partial {\tau ^{(\alpha )}}}}}
            \right|}_t}\Delta {\tau ^{\left( \alpha  \right)}} +\theta {{\left. {\frac{{\partial {{\dot \gamma }^{(\alpha )}}}}
            {{\partial {\alpha ^{(\alpha )}}}}} \right|}_t}\Delta {\alpha ^{\left( \alpha  \right)}} + \theta {{\left.
            {\frac{{\partial {{\dot \gamma }^{(\alpha )}}}}{{\partial {r^{(\alpha )}}}}} \right|}_t}\Delta {r^{\left(
            \alpha  \right)}}} \right]

        令：

        .. math::
            \left\langle X \right\rangle  = \left\langle {\frac{{\left| {{\tau ^{\left( \alpha  \right)}} - {\alpha
            ^{\left( \alpha  \right)}}} \right| - {r^{\left( \alpha  \right)}}}}{{{g^{\left( \alpha  \right)}}}}} \right\rangle

        得到混合强化模型：

        .. math::
            {f^{\left( \alpha  \right)}} = {\rm{sign}}\left( {{\tau ^{\left( \alpha  \right)}} - {\alpha ^{(\alpha )}}}
            \right){\left\langle X \right\rangle ^n}

        接下来采用链式法则，对每一项分别求导：

        .. math::
            \frac{{\partial {f^{\left( \alpha  \right)}}}}{{\partial \left\langle X \right\rangle }} = {\rm{sign}}\left(
            {{\tau ^{\left( \alpha  \right)}} - {\alpha ^{(\alpha )}}} \right)n{\left\langle X \right\rangle ^{n{\rm{ - 1}}}}

        .. math::
            \frac{{\partial \left\langle X \right\rangle }}{{\partial {\tau ^{\left( \alpha  \right)}}}} = \frac{1}{{{g^{
            \left( \alpha  \right)}}}}{\rm{sign}}\left( {{\tau ^{\left( \alpha  \right)}} - {\alpha ^{\left( \alpha  \right)}}} \right) H(X)

        .. math::
            \frac{{\partial \left\langle X \right\rangle }}{{\partial {\alpha ^{\left( \alpha  \right)}}}} =  - \frac{1}{{{g^{\left( \alpha
             \right)}}}}{\rm{sign}}\left( {{\tau ^{\left( \alpha  \right)}} - {\alpha ^{\left( \alpha  \right)}}} \right) H(X)

        .. math::
            \frac{{\partial \left\langle X \right\rangle }}{{\partial {r^{\left( \alpha  \right)}}}} =
            - \frac{1}{{{g^{\left( \alpha  \right)}}}} H(X)

        所以有：

        .. math::
            \frac{{\partial {{\dot \gamma }^{\left( \alpha  \right)}}}}{{\partial {\tau ^{\left( \alpha  \right)}}}} =
            {{\dot a}^{\left( \alpha  \right)}}\frac{{\partial {f^{\left( \alpha  \right)}}}}{{\partial {\tau ^{\left(
            \alpha  \right)}}}}  = {{\dot a}^{\left( \alpha  \right)}}\frac{{\partial {f^{\left( \alpha  \right)}}}}
            {{\partial \left| X \right|}}\frac{{\partial \left| X \right|}}{{\partial {\tau ^{\left( \alpha  \right)}}}}
            = {{\dot a}^{\left( \alpha  \right)}}n{\left\langle X \right\rangle ^{n{\rm{ - 1}}}}\frac{1}{{{g^{\left(
            \alpha  \right)}}}} H(X)

        .. math::
            \frac{{\partial {{\dot \gamma }^{\left( \alpha  \right)}}}}{{\partial {\alpha ^{\left( \alpha  \right)}}}} =
            {{\dot a}^{\left( \alpha  \right)}}\frac{{\partial {f^{\left( \alpha  \right)}}}}{{\partial {\alpha ^{\left(
            \alpha  \right)}}}}  = {{\dot a}^{\left( \alpha  \right)}}\frac{{\partial {f^{\left( \alpha  \right)}}}}
            {{\partial \left| X \right|}}\frac{{\partial \left| X \right|}}{{\partial {\alpha ^{\left( \alpha  \right)}}}}
            = - {{\dot a}^{\left( \alpha  \right)}}n{\left\langle X \right\rangle ^{n{\rm{ - 1}}}}\frac{1}{{{g^{\left(
            \alpha  \right)}}}} H(X)

        .. math::
            \frac{{\partial {{\dot \gamma }^{\left( \alpha  \right)}}}}{{\partial {r^{\left( \alpha  \right)}}}} =
            {{\dot a}^{\left( \alpha  \right)}}\frac{{\partial {f^{\left( \alpha  \right)}}}}{{\partial {r^{\left(
            \alpha  \right)}}}} = {{\dot a}^{\left( \alpha  \right)}}\frac{{\partial {f^{\left( \alpha
            \right)}}}}{{\partial \left| X \right|}}\frac{{\partial \left| X \right|}}{{\partial {r^{\left( \alpha  \right)}}}} =
            =  - {{\dot a}^{\left( \alpha  \right)}}n{\left\langle X \right\rangle ^{n{\rm{ - 1}}}}\frac{1}{{{g^{\left(
            \alpha  \right)}}}}{\rm{sign}}\left( {{\tau ^{\left( \alpha  \right)}} - {\alpha ^{(\alpha )}}} \right) H(X)

        将上面的式子带入式 :math:`\Delta \gamma^{(\alpha)}` ，将每个滑移系 :math:`\alpha` 的剪切应变增量 :math:`\Delta \gamma^{(\alpha)}` 写成矩阵形式，可得：

        .. math::
            \Delta {\gamma ^{\left( \alpha  \right)}} = \Delta t\left[ \begin{array}{l}
            {{\dot \gamma }^{\left( \alpha  \right)}}\left( t \right)\\
             + \theta {{\dot a}^{\left( \alpha  \right)}}n{\left\langle X \right\rangle ^{n{\rm{ - 1}}}}\frac{1}{{{g^{\left( \alpha  \right)}}}} H(X) \Delta {\tau ^{\left( \alpha  \right)}} \\
             - \theta {{\dot a}^{\left( \alpha  \right)}}n{\left\langle X \right\rangle ^{n{\rm{ - 1}}}}\frac{1}{{{g^{\left( \alpha  \right)}}}} H(X) \Delta {\alpha ^{\left( \alpha  \right)}}\\
             - \theta {{\dot a}^{\left( \alpha  \right)}}n{\left\langle X \right\rangle ^{n{\rm{ - 1}}}}\frac{1}{{{g^{\left( \alpha  \right)}}}} H(X) {\rm{sign}}\left( {{\tau ^{\left( \alpha  \right)}} - {\alpha ^{(\alpha )}}} \right)\Delta {r^{\left( \alpha  \right)}}
            \end{array} \right]

        进一步将 :math:`\Delta \tau^{(\alpha)}` ， :math:`\Delta \alpha^{(\alpha)}` 和 :math:`\Delta r^{(\alpha)}`
        的表达式代入剪切应变增量方程 :math:`\Delta \gamma^{(\alpha)}` ，同时注意到 :math:`\langle X\rangle^{n-1} H(X)=\langle X\rangle^{n-1}` ，可以得到：

        .. math::
            \sum\limits_{\beta  = 1}^N {{\delta _{\alpha \beta }}\Delta {\gamma ^{\left( \beta  \right)}}} {\rm{ = }}\Delta t\left\{ \begin{array}{l}
            {{\dot \gamma }^{\left( \alpha  \right)}}\left( t \right)\\
             + \theta {{\dot a}^{\left( \alpha  \right)}}n{\left\langle X \right\rangle ^{n{\rm{ - 1}}}}\frac{1}{{{g^{\left( \alpha  \right)}}}}\left[ {{{\boldsymbol{S}}^{\left( \alpha  \right)}}:\Delta {\boldsymbol{\varepsilon }} - {{\boldsymbol{S}}^{\left( \alpha  \right)}}:\sum\limits_{\beta  = 1}^N {{{\boldsymbol{P}}^{\left( \beta  \right)}}\Delta {\gamma ^{\left( \beta  \right)}}} } \right]\\
             - \theta {{\dot a}^{\left( \alpha  \right)}}n{\left\langle X \right\rangle ^{n{\rm{ - 1}}}}\frac{1}{{{g^{\left( \alpha  \right)}}}}\left( {{c_1}\sum\limits_{\beta  = 1}^N {{\delta _{\alpha \beta }}\Delta {\gamma ^{\left( \beta  \right)}}}  - {c_2}{\rm{sign}}\left( {{{\dot \gamma }^{\left( \beta  \right)}}} \right)\sum\limits_{\beta  = 1}^N {{\delta _{\alpha \beta }}\Delta {\gamma ^{\left( \beta  \right)}}} {\alpha ^{\left( \alpha  \right)}}} \right)\\
             - \theta {{\dot a}^{\left( \alpha  \right)}}n{\left\langle X \right\rangle ^{n{\rm{ - 1}}}}\frac{1}{{{g^{\left( \alpha  \right)}}}}{\rm{sign}}\left( {{\tau ^{\left( \alpha  \right)}} - {\alpha ^{(\alpha )}}} \right)bQ\sum\limits_\beta  {{h_{\alpha \beta }}\left( {1 - b{\rho ^{\left( \beta  \right)}}} \right)\left| {\Delta {\gamma ^{\left( \beta  \right)}}} \right|}
            \end{array} \right\}

        合并式子：

        .. math::
            \begin{aligned}
            \begin{array}{l}
            \sum\limits_{\beta  = 1}^N {\left[ \begin{array}{l}
            {\delta _{\alpha \beta }}\\
             + \Delta t\theta {{\dot a}^{\left( \alpha  \right)}}n{\left\langle X \right\rangle ^{n{\rm{ - 1}}}}\frac{1}{{{g^{\left( \alpha  \right)}}}}{{\boldsymbol{S}}^{\left( \alpha  \right)}}:{{\boldsymbol{P}}^{\left( \beta  \right)}}\\
             + \Delta t\theta {{\dot a}^{\left( \alpha  \right)}}n{\left\langle X \right\rangle ^{n{\rm{ - 1}}}}\frac{1}{{{g^{\left( \alpha  \right)}}}}{\delta _{\alpha \beta }}\left( {{c_1} - {c_2}{\rm{sign}}\left( {{{\dot \gamma }^{\left( \beta  \right)}}} \right){\alpha ^{\left( \alpha  \right)}}} \right)\\
             + \Delta t\theta {{\dot a}^{\left( \alpha  \right)}}n{\left\langle X \right\rangle ^{n{\rm{ - 1}}}}\frac{1}{{{g^{\left( \alpha  \right)}}}}{\rm{sign}}\left( {{\tau ^{\left( \alpha  \right)}} - {\alpha ^{(\alpha )}}} \right)bQ{h_{\alpha \beta }}\left( {1 - b{\rho ^{\left( \beta  \right)}}} \right){\rm{sign}}\left( {{{\dot \gamma }^{\left( \beta  \right)}}} \right)
            \end{array} \right]\Delta {\gamma ^{\left( \beta  \right)}}} \\
             = \Delta t{{\dot \gamma }^{\left( \alpha  \right)}}\left( t \right) + \Delta t\theta {{\dot a}^{\left( \alpha  \right)}}n{\left\langle X \right\rangle ^{n{\rm{ - 1}}}}\frac{1}{{{g^{\left( \alpha  \right)}}}}{{\boldsymbol{S}}^{\left( \alpha  \right)}}:\Delta {\boldsymbol{\varepsilon }}
            \end{array}
            \end{aligned}

        编程过程中为了简化计算，引入一些临时变量：

        .. math::
            term1=\Delta t \theta

        .. math::
            term2 = term1 \cdot \dot{a}^{(\alpha)} n\langle X\rangle^{n-1} \cdot \frac{1}{{{g^{\left( \alpha  \right)}}}}

        .. math::
            term4=\boldsymbol{S}^{(\alpha)}: \boldsymbol{P}^{(\beta)}

        整理得到引入线性背应力项（随动强化项）和各向同性强化模型的迭代格式为由 N 个末知数 :math:`\Delta \gamma^{(\beta)}` 和 N 个非线性方程组成的方程组：

        .. math::
            \begin{aligned}
            & \left[\begin{array}{l}
            \delta_{\alpha \beta} \\
            +term2 \cdot term4\\
            + term2 \cdot {\delta _{\alpha \beta }}\left( {{c_1} - {c_2}{\rm{sign}}\left( {{{\dot \gamma }^{\left( \beta  \right)}}} \right){\alpha ^{\left( \alpha  \right)}}} \right)\\
            {\rm{ + }}term2 \cdot {\rm{sign}}\left( {{\tau ^{\left( \alpha  \right)}} - {\alpha ^{(\alpha )}}} \right) \cdot bQ{h_{\alpha \beta }}\left( {1 - b{\rho ^{\left( \beta  \right)}}} \right) \cdot {\rm{sign}}\left( {{{\dot \gamma }^{\left( \beta  \right)}}} \right)
            \end{array}\right] \Delta \gamma^{(\alpha)} \\
            & =\Delta t \dot{\gamma}^{(\alpha)}(t)+term2 \cdot \rm \boldsymbol {S}^{(\alpha)}: \Delta {\boldsymbol{\varepsilon }}
            \end{aligned}

        其中，上标 :math:`\alpha` 表示第 :math:`\alpha` 个滑移系( :math:`\alpha=1 \sim N，N` 为所有可能开动滑移系的数目)，
        等式右边的 :math:`\Delta {\boldsymbol{\varepsilon }}` 为已知项。求解该非线性方程组可以得到所有滑移系的初始剪切应变增量 :math:`\Delta \gamma^{(\alpha)}`，
        进而计算应力增量 :math:`\Delta \sigma` 和其他状态变量的初始增量。

        将上面的方程组简写为以下形式：

        .. math::
            {{\mathbf{A}}} \Delta {\gamma ^{\left( \alpha  \right)}} = \Delta t{{\dot \gamma }^{\left( \alpha  \right)}}\left( t \right)
            + term2 \cdot {{\boldsymbol{S}}^{\left( \alpha  \right)}}:\Delta {\boldsymbol{\varepsilon }}

        方程两边对 :math:`\Delta {\boldsymbol{\varepsilon }}` 求偏导得到：

        .. math::
            {{\mathbf{A}}} \frac{{\partial \Delta {\gamma ^{\left( \alpha  \right)}}}}{{\partial \Delta {\boldsymbol{\varepsilon }}}} =
            term2 \cdot {{\boldsymbol{S}}^{\left( \alpha  \right)}}

        所以有：

        .. math::
            ddgdde = \frac{{\partial \Delta {\gamma ^{\left( \alpha  \right)}}}}{{\partial \Delta {\boldsymbol{\varepsilon }}}} =
            {{{\mathbf{A}}}^{ - 1}} \cdot term2 \cdot {{\boldsymbol{S}}^{\left( \alpha  \right)}}

        进而，我们可以得到，弹性模量张量的弹性部分，即  :math:`ddsdde` 矩阵为：

        .. math::
            ddsdde = \mathbb{C} - {{\boldsymbol{S}}^{\left( \alpha  \right)}} \cdot ddgdde

        3.2.2. 迭代求解剪切应变增量以及更新切线刚度矩阵

        采用牛顿拉夫森迭代方法进行迭代求解。在上面的推导中，由于 :math:`{{\dot \gamma }^{\left( \alpha  \right)}}` 使用泰勒展开略去了高阶小量，
        导致初始剪切应变增量 :math:`\Delta \gamma^{(\alpha)}` 产生了误差Residual。

        我们可以写出Residual的表达式：

        .. math::
            Residual = F\left( {\Delta {\gamma ^{\left( \alpha  \right)}}} \right) = \Delta {\gamma ^{\left( \alpha  \right)}} -
            \Delta t\left( {1 - \theta } \right){{\dot \gamma }^{\left( \alpha  \right)}}\left( t \right) -
            \Delta t\theta {{\dot \gamma }^{\left( \alpha  \right)}}\left( {t + \Delta t} \right)

        其中， :math:`\Delta \gamma^{(\alpha)}` 是我们利用 3.2.1 节的非线性方程组求出的近似值，即初值。同时，上式也是牛顿拉弗森法迭代的目标函数。
        我们要做的就是对这个函数上的点做切线，并求切线的零点。即使得Residual为 0 或接近我们的预设阈值 tolerance ，可用数学式表达为：

        .. math::
            F'\left( {{{\left\{ {\Delta {\gamma ^{\left( \alpha  \right)}}} \right\}}^{\left( k \right)}}} \right)
            \cdot \Delta {\left\{ {\Delta {\gamma ^{\left( \alpha  \right)}}} \right\}^{\left( {k + 1} \right)}} =
            0 - F\left( {{{\left\{ {\Delta {\gamma ^{\left( \alpha  \right)}}} \right\}}^{\left( k \right)}}} \right)

        其中，

        .. math::
            \Delta {\left\{ {\Delta {\gamma ^{\left( \alpha  \right)}}} \right\}^{\left( {k + 1} \right)}} =
            {\left\{ {\Delta {\gamma ^{\left( \alpha  \right)}}} \right\}^{\left( {k + 1} \right)}} - {\left\{ {\Delta
            {\gamma ^{\left( \alpha  \right)}}} \right\}^{\left( k \right)}}

        当初值计算完成后，我们获得了新的分解剪应力 :math:`\tau_{t+\Delta t}^{\alpha}` ，背应力 :math:`\alpha_{t+\Delta t}^{\alpha}`
        和各项同性强化项 :math:`r_{t+\Delta t}^{\alpha}` ，从而我们获得了新的 :math:`X` ：

        .. math::
            X = \frac{{\left| {\tau _t^{\left( \alpha  \right)} + \Delta {\tau ^{\left( \alpha  \right)}} - \left(
            {\alpha _t^{\left( \alpha  \right)} + \Delta {\alpha ^{\left( \alpha  \right)}}} \right)} \right| -
            \left( {r_t^{\left( \alpha  \right)} + \Delta {r^{\left( \alpha  \right)}}} \right)}}{{{g^{\left( \alpha  \right)}}}}

        进而可得到即将用于牛顿拉夫森迭代的剪切应变速率表达式：

        .. math::
            \dot \gamma _{t + \Delta t}^{\left( \alpha  \right)} = {{\dot a}^{\left( \alpha  \right)}}{\left\langle X
            \right\rangle ^n}{\text{sign}}\left( {\tau _t^{\left( \alpha  \right)} + \Delta {\tau ^{\left( \alpha
            \right)}} - \left( {\alpha _t^{\left( \alpha  \right)} + \Delta {\alpha ^{\left( \alpha  \right)}}} \right)} \right)

        参考3.2.1节的推导，最后我们得到迭代求解该非线性方程组的所有滑移系的剪切应变增量的方程组：

        .. math::
            {{\mathbf{A}}_{1}} \cdot \Delta {\left\{ {\Delta {\gamma ^{\left( \alpha  \right)}}} \right\}^{\left( {k + 1} \right)}} = rhs

        其中，刚度矩阵 :math:`{{\mathbf{A}}_{1}}` 为：

        .. math::
            {{\mathbf{A}}_{1}} = F'\left( {\Delta {\gamma ^{\left( \alpha  \right)}}} \right) = \left[ {\begin{array}{*{20}{l}}
              {{\delta _{\alpha \beta }}} \\
              {term2 \cdot term4} \\
              { + term2 \cdot {\delta _{\alpha \beta }}\left( {{c_1} - {c_2}{\alpha ^{(\alpha )}}{\text{sign}}\left( {\Delta {\gamma ^{(\alpha )}}} \right)} \right)} \\
              { + term2 \cdot bQ{h_{\alpha \beta }}\left( {1 - b{\rho ^{(\beta )}}} \right){\delta _{\alpha \beta }}{\text{sign}}\left( {\Delta {\gamma ^{(\alpha )}}} \right){\text{sign}}\left( {\tau _t^{\left( \alpha  \right)} + \Delta {\tau ^{\left( \alpha  \right)}} - \left( {\alpha _t^{\left( \alpha  \right)} + \Delta {\alpha ^{\left( \alpha  \right)}}} \right)} \right)}
            \end{array}} \right]

        式中，

        .. math::
            term1=\Delta t \theta

        .. math::
            term2 = term1 \cdot \dot{a}^{(\alpha)} n\langle X\rangle^{n-1} \cdot \frac{1}{{{g^{\left( \alpha  \right)}}}}

        .. math::
            term4=\boldsymbol{S}^{(\alpha)}: \boldsymbol{P}^{(\beta)}

        方程组右边项 :math:`rhs` 为：

        .. math::
            rhs =  - F\left( {{{\left\{ {\Delta {\gamma ^{\left( \alpha  \right)}}} \right\}}^{\left( k \right)}}} \right)
            = \Delta t\left( {1 - \theta } \right){{\dot \gamma }^{\left( \alpha  \right)}}\left( t \right) +
            \Delta t\theta {{\dot \gamma }^{\left( \alpha  \right)}}\left( {t + \Delta t} \right) - \Delta {\gamma ^{\left( \alpha  \right)}}

        可以看出，若

        .. math::
            {{\text{sign}}\left( {\Delta {\gamma ^{(\alpha )}}} \right)} = {{\text{sign}}\left( {{{\dot \gamma }^{\left( \beta  \right)}}} \right)},
            {\text{sign}}\left( {{\tau ^{\left( \alpha  \right)}} - {\alpha ^{(\alpha )}}} \right) = {{\text{sign}}\left( {\tau _t^{\left( \alpha
            \right)} + \Delta {\tau ^{\left( \alpha  \right)}} - \left( {\alpha _t^{\left( \alpha  \right)} + \Delta {\alpha ^{\left( \alpha  \right)}}} \right)} \right)}

        则刚度矩阵 :math:`{{\mathbf{A}}_{1}}` 与求解初值的刚度矩阵 :math:`{\mathbf{A}}` 形式一致。

        下面列出本构模型中的变量或常数与程序中变量名或常数的对应关系：

        应力 :math:`\sigma` ：stress

        应力增量：delta_stress

        应变 :math:`\varepsilon` ：strain

        应变增量 :math:`\Delta \varepsilon` :dstrain

        应变增量 :math:`\Delta \varepsilon^{e}` ：delta_elastic_strain

        分解剪应力 :math:`\tau` ：tau

        状态变量 :math:`\rho` ：rho

        背应力项(随动强化项) :math:`\alpha` ：alpha

        各向同性强化项 :math:`r` ：r

        滑移系滑移方向的单位向量 :math:`{\boldsymbol{m}}` 和 :math:`{\boldsymbol{m}}^{*}` ：m_s

        滑移系滑移面法向的单位向量 :math:`{\boldsymbol{n}}` 和 :math:`{\boldsymbol{n}}^{*}` ：n_s

        塑性旋率张量 :math:`{{\boldsymbol{W}}^{\rm{p}}}` 中的 :math:`\boldsymbol{\Omega}^{(\alpha)}` ：Omega

        塑性变形率张量 :math:`{{\boldsymbol{D}}^{\rm{p}}}` 中的 :math:`{\boldsymbol{P}}^{\left( \alpha  \right)}` ：P

        求解Jaumann率的中间项 :math:`{{\boldsymbol {\Omega }}^{\left( \alpha  \right)}} \cdot {\boldsymbol{\sigma}} -
        {\boldsymbol{\sigma}} \cdot {{\rm{\Omega }}^{\left( \alpha  \right)}}` ：B

        Jaumann率中的旋转部分 :math:`\mathbb{C}:{{\boldsymbol{P}}^{\left( \alpha  \right)}} +  {{\boldsymbol {\Omega }}^{\left(
        \alpha \right)}} \cdot {\boldsymbol{\sigma}} - {\boldsymbol{\sigma}} \cdot {{\rm{\Omega }}^{\left( \alpha  \right)}}` ：S

        弹性模量张量 :math:`\mathbb{C}` ：C

        剪切应变 :math:`\gamma` ：gamma

        滑移系的剪切应变率 :math:`\dot{\gamma}` ：gamma_dot

        滑移系的应变速率敏感指数 :math:`n` ：p_s

        滑移系的参考剪切应变率 :math:`\dot{a}` ：dot_gamma_0

        参考屈服强度 :math:`g^{(\alpha)}` ：K



        剪切应变速率初值 :math:`\Delta \gamma_{t}^{(\alpha)}` ：gamma_dot_t

        用于迭代的剪切应变速率 :math:`\Delta \gamma_{t+\Delta t}^{(\alpha)}` ：gamma_dot

        混合强化模型的中间变量 :math:`X` ：X

        背应力项参数 :math:`c_{1}` ：c_1

        背应力项参数 :math:`c_{2}` ：c_2

        各向同性强化项的临界分解剪应力(初始阻应力) :math:`r_{0}` ：r_0

        各向同性强化项参数 :math:`b` ：b

        各向同性强化项参数 :math:`Q`  ：Q

        参考文献：

        [1]. Nonlinear Finite Elements for Continua and Structures, Ted Belytschko.

        """
        strain = variable['strain']
        dstrain = variable['dstrain']

        if self.section.type == 'PlaneStrain':
            strain = array([strain[0], strain[1], 0.0, strain[2], 0.0, 0.0])
            dstrain = array([dstrain[0], dstrain[1], 0.0, dstrain[2], 0.0, 0.0])

        np.set_printoptions(precision=12, linewidth=256, suppress=True)

        K = self.K
        dot_gamma_0 = self.dot_gamma_0
        p_s = self.p_s
        dt = timer.dtime
        theta = self.theta
        c_1 = self.c_1
        c_2 = self.c_2
        r_0 = self.r_0
        b = self.b
        Q = self.Q
        H = self.H
        C = self.C
        m_s = self.m_s
        n_s = self.n_s

        if state_variable == {} or timer.time0 == 0.0:
            state_variable['m_s'] = m_s
            state_variable['n_s'] = n_s
            m_sxn_s = transpose(array([m_s[:, 0] * n_s[:, 0],
                                       m_s[:, 1] * n_s[:, 1],
                                       m_s[:, 2] * n_s[:, 2],
                                       2.0 * m_s[:, 0] * n_s[:, 1],
                                       2.0 * m_s[:, 0] * n_s[:, 2],
                                       2.0 * m_s[:, 1] * n_s[:, 2]]))
            n_sxm_s = transpose(array([n_s[:, 0] * m_s[:, 0],
                                       n_s[:, 1] * m_s[:, 1],
                                       n_s[:, 2] * m_s[:, 2],
                                       2.0 * n_s[:, 0] * m_s[:, 1],
                                       2.0 * n_s[:, 0] * m_s[:, 2],
                                       2.0 * n_s[:, 1] * m_s[:, 2]]))
            P = 0.5 * (m_sxn_s + n_sxm_s)
            state_variable['stress'] = zeros(shape=6, dtype=DTYPE)
            state_variable['tau'] = dot(P, state_variable['stress'])
            state_variable['gamma'] = zeros(shape=self.total_number_of_slips, dtype=DTYPE)
            state_variable['rho'] = zeros(shape=self.total_number_of_slips, dtype=DTYPE)
            state_variable['alpha'] = zeros(shape=self.total_number_of_slips, dtype=DTYPE)
            state_variable['r'] = zeros(shape=self.total_number_of_slips, dtype=DTYPE) + r_0

        rho = deepcopy(state_variable['rho'])
        m_s = deepcopy(state_variable['m_s'])
        n_s = deepcopy(state_variable['n_s'])
        gamma = deepcopy(state_variable['gamma'])
        stress = deepcopy(state_variable['stress'])
        tau = deepcopy(state_variable['tau'])
        alpha = deepcopy(state_variable['alpha'])
        r = deepcopy(state_variable['r'])

        delta_gamma = zeros(shape=self.total_number_of_slips, dtype=DTYPE)

        is_convergence = False

        for niter in range(self.MAX_NITER):
            m_sxn_s = transpose(array([m_s[:, 0] * n_s[:, 0],
                                       m_s[:, 1] * n_s[:, 1],
                                       m_s[:, 2] * n_s[:, 2],
                                       2.0 * m_s[:, 0] * n_s[:, 1],
                                       2.0 * m_s[:, 0] * n_s[:, 2],
                                       2.0 * m_s[:, 1] * n_s[:, 2]]))

            n_sxm_s = transpose(array([n_s[:, 0] * m_s[:, 0],
                                       n_s[:, 1] * m_s[:, 1],
                                       n_s[:, 2] * m_s[:, 2],
                                       2.0 * n_s[:, 0] * m_s[:, 1],
                                       2.0 * n_s[:, 0] * m_s[:, 2],
                                       2.0 * n_s[:, 1] * m_s[:, 2]]))

            P = 0.5 * (m_sxn_s + n_sxm_s)
            Omega = 0.5 * (m_sxn_s - n_sxm_s)
            Omega[:, 3:] *= 0.5

            # B = Omega * stress - stress * Omega
            # S = dot(P, C) + B
            S = dot(P, C)

            X = (abs(tau - alpha) - r) / K
            gamma_dot = dot_gamma_0 * maximum(X, 0.0) ** p_s * sign(tau - alpha)

            if niter == 0:
                gamma_dot_t = deepcopy(gamma_dot)

            term1 = dt * theta
            term2 = term1 * dot_gamma_0 * p_s * maximum(X, 0.0) ** (p_s - 1.0) / K
            term3 = term1 * maximum(X, 0) * dot_gamma_0 * p_s * maximum(X, 0.0) ** (p_s - 1.0) / K
            term4 = einsum('ik, jk->ij', S, P)

            A = eye(self.total_number_of_slips, dtype=DTYPE)
            A += term2 * term4
            # A += H * term3 * sign(gamma_dot) * sign(tau - alpha)
            A += term2 * (c_1 - c_2 * alpha * sign(gamma_dot)) * eye(self.total_number_of_slips, dtype=DTYPE)
            A += term2 * b * Q * H * (1.0 - b * rho) * sign(tau - alpha) * sign(gamma_dot)

            if niter == 0:
                rhs = dt * gamma_dot + term2 * dot(S, dstrain)
            else:
                rhs = term1 * (gamma_dot - gamma_dot_t) + gamma_dot_t * dt - delta_gamma

            d_delta_gamma = solve(transpose(A), rhs)
            delta_gamma += d_delta_gamma

            delta_elastic_strain = dstrain - dot(delta_gamma, P)
            delta_tau = dot(S, delta_elastic_strain)
            delta_stress = dot(C, delta_elastic_strain)
            delta_alpha = c_1 * delta_gamma - c_2 * abs(delta_gamma) * alpha
            delta_rho = (1.0 - b * rho) * abs(delta_gamma)
            delta_r = b * Q * dot(H, delta_rho)

            gamma = deepcopy(state_variable['gamma']) + delta_gamma
            tau = deepcopy(state_variable['tau']) + delta_tau
            stress = deepcopy(state_variable['stress']) + delta_stress
            alpha = deepcopy(state_variable['alpha']) + delta_alpha
            rho = deepcopy(state_variable['rho']) + delta_rho
            r = deepcopy(state_variable['r']) + delta_r

            X = (abs(tau - alpha) - r) / K
            gamma_dot = dot_gamma_0 * maximum(X, 0.0) ** p_s * sign(tau - alpha)
            residual = dt * theta * gamma_dot + dt * (1.0 - theta) * gamma_dot_t - delta_gamma

            # if element_id == 0 and iqp == 0:
            #     print('residual', residual)

            if all(residual < self.tolerance):
                is_convergence = True
                break

        ddgdde = term2.reshape((self.total_number_of_slips, 1)) * S
        ddgdde = dot(inv(A), ddgdde)
        ddsdde = C - einsum('ki, kj->ij', S, ddgdde)

        if not is_convergence:
            timer.is_reduce_dtime = True

        state_variable_new['m_s'] = m_s
        state_variable_new['n_s'] = n_s
        state_variable_new['stress'] = stress
        state_variable_new['gamma'] = gamma
        state_variable_new['tau'] = tau
        state_variable_new['alpha'] = alpha
        state_variable_new['r'] = r
        state_variable_new['rho'] = rho

        some_energy = 0.5 * sum(strain * stress)

        if self.section.type == 'PlaneStrain':
            ddsdde = delete(delete(ddsdde, [2, 4, 5], axis=0), [2, 4, 5], axis=1)
            stress = delete(stress, [2, 4, 5])

        output = {'stress': stress, 'plastic_energy': some_energy}

        return ddsdde, output


if __name__ == "__main__":
    from pyfem.utils.visualization import print_slots_dict

    print_slots_dict(PlasticCrystal.__slots_dict__)

    from pyfem.Job import Job

    # job = Job(r'..\..\..\examples\mechanical\1element\hex20_crystal\Job-1.toml')
    job = Job(r'..\..\..\examples\mechanical\4_grains_crystal\Job-1.toml')

    job.run()