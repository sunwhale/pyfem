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

    :ivar c_1: 随动强化参数
    :vartype c_1: ndarray

    :ivar c_2: 随动强化参数
    :vartype c_2: ndarray

    :ivar r_0: 随动强化参数
    :vartype r_0: ndarray

    :ivar b_s: 各向同性强化参数
    :vartype b_s: ndarray

    :ivar Q_s: 各项同性强化参数
    :vartype Q_s: ndarray

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

    :ivar T_voigt: Vogit坐标变换矩阵
    :vartype T_voigt: ndarray

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
        'b_s': ('ndarray', '各项同性强化参数'),
        'Q_s': ('ndarray', '各项同性强化参数'),
        'H': ('ndarray', '硬化系数矩阵'),
        'u_global': ('ndarray', '全局坐标系下的1号矢量'),
        'v_global': ('ndarray', '全局坐标系下的2号矢量'),
        'w_global': ('ndarray', '全局坐标系下的3号矢量'),
        'u_grain': ('ndarray', '晶粒坐标系下的1号矢量'),
        'v_grain': ('ndarray', '晶粒坐标系下的2号矢量'),
        'w_grain': ('ndarray', '晶粒坐标系下的3号矢量'),
        'T': ('ndarray', '坐标变换矩阵'),
        'T_voigt': ('ndarray', 'Vogit坐标变换矩阵'),
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
            b_s = ones((slip_system_number,), dtype=DTYPE) * material.data_dict['b_s'][i]
            Q_s = ones((slip_system_number,), dtype=DTYPE) * material.data_dict['Q_s'][i]
            if i == 0:
                self.m_s: ndarray = m_s
                self.n_s: ndarray = n_s
                self.K: ndarray = K
                self.dot_gamma_0: ndarray = dot_gamma_0
                self.p_s: ndarray = p_s
                self.c_1: ndarray = c_1
                self.c_2: ndarray = c_2
                self.r_0: ndarray = r_0
                self.b_s: ndarray = b_s
                self.Q_s: ndarray = Q_s
            else:
                self.m_s = concatenate((self.m_s, m_s))
                self.n_s = concatenate((self.n_s, n_s))
                self.K = concatenate((self.K, K))
                self.dot_gamma_0 = concatenate((self.dot_gamma_0, dot_gamma_0))
                self.p_s = concatenate((self.p_s, p_s))
                self.c_1 = concatenate((self.c_1, c_1))
                self.c_2 = concatenate((self.c_2, c_2))
                self.r_0 = concatenate((self.r_0, r_0))
                self.b_s = concatenate((self.b_s, b_s))
                self.Q_s = concatenate((self.Q_s, Q_s))

        self.H = ones(shape=(self.total_number_of_slips, self.total_number_of_slips), dtype=DTYPE)

        # 晶粒取向信息
        self.u_global: ndarray = array(section.data_dict['u_global'])
        self.v_global: ndarray = array(section.data_dict['v_global'])
        self.w_global: ndarray = array(section.data_dict['w_global'])

        self.u_grain: ndarray = array(section.data_dict['u_grain'])
        self.v_grain: ndarray = array(section.data_dict['v_grain'])
        self.w_grain: ndarray = array(section.data_dict['w_grain'])

        self.T: ndarray = get_transformation(self.u_grain, self.v_grain, self.w_grain, self.u_global, self.v_global, self.w_global)
        self.T_voigt: ndarray = get_voigt_transformation(self.T)

        # 旋转至全局坐标系
        self.m_s = dot(self.m_s, self.T)
        self.n_s = dot(self.n_s, self.T)
        self.C = dot(dot(self.T_voigt, self.C), transpose(self.T_voigt))

    def create_elastic_stiffness(self, elastic: dict):
        r"""
        **定义局部晶系的弹性刚度矩阵**

        弹性刚度矩阵由弹性常数组成，对应的矩阵形式与弹性常数个数及材料对称性相关，相关参数由材料属性数据字典中的 :py:attr:`elastic` 字典给出。

        （1）各向同性材料(Isotropic material)：对于一般各向同性材料，其包含两个独立的弹性常数，即：杨氏模量(Young's modulus) :math:`E` 和泊松比(Poisson's ratio) :math:`\nu` ,
        进一步可得到这两个弹性常数与剪切模量 :math:`G = \mu` 和拉梅常数 :math:`\lambda` 的关系为([1],[2])：

        .. math::
            \lambda  = \frac{{\nu E}}{{(1 + \nu )(1 - 2\nu )}},G = \mu  = \frac{E}{{2(1 + \nu )}}

        进而得到各向同性材料得到弹性矩阵形式为：

        .. math::
            {{\mathbf{C}}_{(2D)}} = \left[ {\begin{array}{*{20}{c}}
              {\lambda  + 2\mu }&\lambda &0 \\
              \lambda &{\lambda  + 2\mu }&0 \\
              0&0&\mu
            \end{array}} \right]

        .. math::
            {{\mathbf{C}}_{(3D)}} = \left[ {\begin{array}{*{20}{c}}
              {\lambda  + 2\mu }&\lambda &\lambda &0&0&0 \\
              \lambda &{\lambda  + 2\mu }&\lambda &0&0&0 \\
              \lambda &\lambda &{\lambda  + 2\mu }&0&0&0 \\
              0&0&0&\mu &0&0 \\
              0&0&0&0&\mu &0 \\
              0&0&0&0&0&\mu
            \end{array}} \right]

        （2）立方材料(Cubic material)：包含三个独立的材料参数 :math:`{C_{11}},{C_{12}},{C_{44}}` ，其弹性矩阵定义为：

        .. math::
            {{\mathbf{C}}_{(2D)}} = \left[ {\begin{array}{*{20}{c}}
              {{C_{11}}}&{{C_{12}}}&0 \\
              {{C_{12}}}&{{C_{11}}}&0 \\
              0&0&{{C_{44}}}
            \end{array}} \right]

        .. math::
            {{\mathbf{C}}_{(3D)}} = \left[ {\begin{array}{*{20}{c}}
              {{C_{11}}}&{{C_{12}}}&{{C_{12}}}&0&0&0 \\
              {{C_{12}}}&{{C_{11}}}&{{C_{12}}}&0&0&0 \\
              {{C_{12}}}&{{C_{12}}}&{{C_{11}}}&0&0&0 \\
              0&0&0&{{C_{44}}}&0&0 \\
              0&0&0&0&{{C_{44}}}&0 \\
              0&0&0&0&0&{{C_{44}}}
            \end{array}} \right]

        （3）正交材料(Orthotropic material)：包含9个独立的材料参数，分别为：

         .. math::
            {C_{1111}},{C_{1122}},{C_{2222}},{C_{1133}},{C_{2233}},{C_{3333}},{C_{1212}},{C_{1313}},{C_{2323}}

        与 ABAQUS 对各向同性材料的定义相同，其弹性矩阵定义为：

        .. math::
            {{\mathbf{C}}_{(2D)}} = \left[ {\begin{array}{*{20}{c}}
              {{C_{1111}}}&{{C_{1122}}}&0 \\
              {{C_{1122}}}&{{C_{2222}}}&0 \\
              0&0&{{C_{1212}}}
            \end{array}} \right]

        .. math::
            {{\mathbf{C}}_{(3D)}} = \left[ {\begin{array}{*{20}{c}}
              {{C_{1111}}}&{{C_{1122}}}&{{C_{1133}}}&0&0&0 \\
              {{C_{1122}}}&{{C_{2222}}}&{{C_{2233}}}&0&0&0 \\
              {{C_{1133}}}&{{C_{2233}}}&{{C_{3333}}}&0&0&0 \\
              0&0&0&{{C_{1212}}}&0&0 \\
              0&0&0&0&{{C_{1313}}}&0 \\
              0&0&0&0&0&{{C_{2323}}}
            \end{array}} \right]

        （4）各向异性材料(Anistropic material)：包含21个独立的材料参数，分别为：

        .. math::
            {C_{1111}},{C_{1122}},{C_{2222}},{C_{1133}},{C_{2233}},{C_{3333}},{C_{1112}}

        .. math::
            {C_{2212}},{C_{3312}},{C_{1212}},{C_{1113}},{C_{2213}},{C_{3313}},{C_{1213}}

        .. math::
            {C_{1313}},{C_{1123}},{C_{2223}},{C_{3323}},{C_{1223}},{C_{1323}},{C_{2323}}

        与 ABAQUS 对各向异性材料的定义相同，其弹性矩阵定义为：

        .. math::
            {{\mathbf{C}}_{(2D)}} = \left[ {\begin{array}{*{20}{c}}
              {{C_{1111}}}&{{C_{1122}}}&0 \\
              {{C_{1122}}}&{{C_{2222}}}&0 \\
              0&0&{{C_{1212}}}
            \end{array}} \right]

        .. math::
            {{\mathbf{C}}_{(3D)}} = \left[ {\begin{array}{*{20}{c}}
              {{C_{1111}}}&{{C_{1122}}}&{{C_{1133}}}&{{C_{1112}}}&{{C_{1113}}}&{{C_{1123}}} \\
              {{C_{1122}}}&{{C_{2222}}}&{{C_{2233}}}&{{C_{2212}}}&{{C_{2213}}}&{{C_{2223}}} \\
              {{C_{1133}}}&{{C_{2233}}}&{{C_{3333}}}&{{C_{3312}}}&{{C_{3313}}}&{{C_{3323}}} \\
              {{C_{1112}}}&{{C_{2212}}}&{{C_{3312}}}&{{C_{1212}}}&{{C_{1213}}}&{{C_{1223}}} \\
              {{C_{1113}}}&{{C_{2213}}}&{{C_{3313}}}&{{C_{1213}}}&{{C_{1313}}}&{{C_{1323}}} \\
              {{C_{1123}}}&{{C_{2223}}}&{{C_{3323}}}&{{C_{1223}}}&{{C_{1323}}}&{{C_{2323}}}
            \end{array}} \right]

        [1] I.S. Sokolnikoff: Mathematical Theory of Elasticity. New York, 1956.

        [2] T.J.R. Hughes: The Finite Element Method, Linear Static and Dynamic Finite Element Analysis. New Jersey, 1987.

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
        **获得幂指数形式的晶体塑性本构模型**

        本模块中包含3个字典：:py:attr:`variable` ， :py:attr:`state_variable` ， :py:attr:`state_variable_new` 。

        其中，字典 :py:attr:`variable` 存储自由度相关的变量，如应变 :math:`\varepsilon` 和应变增量 :math:`\Delta \varepsilon` 。

        字典 :py:attr:`state_variable` 存储迭代过程中上一个收敛增量步 :math:`t` 时刻的状态变量，如应力 :math:`\sigma` 、分解剪应力 :math:`\tau` 、
        剪切应变 :math:`\gamma` 、状态变量 :math:`\rho` 、背应力项（随动强化项） :math:`\alpha` 、各向同性强化项 :math:`r` 、
        特征滑移系滑移方向 :math:`m\_s` 、特征滑移系滑移面法向 :math:`n\_s` 。这些状态变量在当前增量步 :math:`t+\Delta t` 计算收敛之前是不被更新的。

        字典 :py:attr:`state_variable_new` 存储当前增量步 :math:`t+\Delta t` 时刻的某个中间迭代步 :math:`k` 的状态变量。

        ========================================
        幂指数形式的晶体塑性本构模型
        ========================================

        ----------------------------------------
        1. 引言
        ----------------------------------------

        在晶体塑性本构中通常采用增量形式的本构方程，若全量形式的本构方程采用 :math:`\boldsymbol{\sigma}= \mathbb{C}:{{\boldsymbol{\varepsilon}}}` ，
        其中， :math:`\boldsymbol{\sigma}` 为 Cauchy 应力张量， :math:`\mathbb{C}` 为弹性模量张量， :math:`{{\boldsymbol{\varepsilon}}}` 为应变张量。
        对全量形式的本构方程求时间导数，可得到增量形式的本构方程 :math:`{\boldsymbol{\dot \sigma}}= \mathbb{C}:{{\boldsymbol{D}}}` ，
        其中， :math:`{\boldsymbol{\dot \sigma}}` 为Cauchy应力张量率， :math:`{{\boldsymbol{D}}}` 为变形率张量。
        可以证明，上述增量形式的本构方程中 :math:`{{\boldsymbol{D}}}` 是客观张量，而 :math:`\dot {\boldsymbol{\sigma}}` 不是客观张量。

        要建立正确的材料本构关系必须遵守“客观性公理”这一基本前提。所谓客观性就是物质的力学性质与观察者无关，客观性又称为“标架无差异性”。遵守“客观性公理”的张量称为客观张量，
        也可以定义为在时空变换中保持不变的张量，亦称“时空无差异张量”。用公式可以直接定义满足如下变换率的张量[1]：

        .. math::
            {\boldsymbol{\dot \Lambda }}^{*} = {\boldsymbol{ Q }}{\boldsymbol{\dot \Lambda }}{\boldsymbol{Q}^{T}}

        为客观性张量。上式表示的是对张量的正交变换或时空变换， :math:`\boldsymbol{Q}` 是时空变换矩阵。

        我们知道 Cauchy 应力张量 :math:`\boldsymbol{\sigma}` 是客观张量，即满足：

        .. math::
            {{{\boldsymbol{\sigma }}}^*} = {\boldsymbol{\dot Q }}{\boldsymbol{\sigma }}{\boldsymbol{ Q }^{T}},
            {\boldsymbol{\dot Q }}{\boldsymbol{ Q }^{T}} = {\boldsymbol{I }}

        对 :math:`{{{\boldsymbol{\sigma }}}^*}` 求时间导数：

        .. math::
            {{{\boldsymbol{\dot \sigma }}}^*} = {\boldsymbol{\dot Q }}{\boldsymbol{\sigma }}{\boldsymbol{ Q }^{T}} +
            {\boldsymbol{ Q }}{\boldsymbol{\dot \sigma }}{\boldsymbol{ Q }^{T}} +
            {\boldsymbol{ Q }}{\boldsymbol{\sigma }}{\boldsymbol{\dot Q }^{T}}


        上面的时间导数表明，一般情况下， :math:`{{{\boldsymbol{\dot \sigma }}}^*}` 并不是客观时间导数，由于 :math:`{{{\boldsymbol{\dot \sigma }}}^*} \ne {\boldsymbol{ Q }}{\boldsymbol{\dot \sigma }}{\boldsymbol{ Q }^{T}}` ，
        只有当转动率为零， :math:`{\boldsymbol{\dot Q }} {\text{ = }} 0` ，也就是当 :math:`{\boldsymbol{ Q }} {\text{ = }}` 常量时， :math:`{{{\boldsymbol{\dot \sigma }}}^*}` 才满足Truesdell客观性的要求。

        同理，可证明变形率 :math:`{{\boldsymbol{D}}}` 满足Truesdell客观性的要求。已知变形率 :math:`{{\boldsymbol{D}}}` 可以表示为：

        .. math::
            {\boldsymbol{D}} = \frac{{\boldsymbol{L}} + {\boldsymbol{L}}^{T}}{2}

        式中， :math:`{\boldsymbol{L}}` 是速度梯度， :math:`{\boldsymbol{L}}^{T}` 是速度梯度的转置。对变形率 :math:`{{\boldsymbol{D}}}` 做时空变换：

        .. math::
            {{{\boldsymbol{D}}}^*} = \frac{{{{\boldsymbol{L}}}^*} + {{{\boldsymbol{L}}}^{T*}}}{2}

        其中， :math:`{{\boldsymbol{L}}^{*}}` 是速度梯度的时空变换律， :math:`{{\boldsymbol{L}}^{T*}}` 是速度梯度转置的时空变换律，两者可表示为：

        .. math::
            {{{\boldsymbol{L}}}^*} = {\boldsymbol{ Q }}{\boldsymbol{L }}{\boldsymbol{ Q }^{T}} + {\boldsymbol{\Omega }}

        .. math::
            {{{\boldsymbol{L}}}^{T*}} = {\boldsymbol{ Q }}{\boldsymbol{L }}{\boldsymbol{ Q }^{T}} - {\boldsymbol{\Omega }},

        式中， :math:`{\boldsymbol{\Omega }}={\boldsymbol{ \dot Q }}{\boldsymbol{ Q }^{T}}` 是一反对称张量。从上式可以看出，
        速度梯度 :math:`{\boldsymbol{L}}` 不是客观张量。将其 :math:`{{{\boldsymbol{L}}}^*}` 和 :math:`{{{\boldsymbol{L}}}^{T*}}`
        代入变形率 :math:`{{\boldsymbol{D}}}` 的时空变换律，我们得到：

        .. math::
            {{{\boldsymbol{D}}}^*} = \frac{{{{\boldsymbol{L}}}^*} + {{{\boldsymbol{L}}}^{T*}}}{2} =
            {\boldsymbol{ Q }}\frac{{{{\boldsymbol{L}}}} + {{{\boldsymbol{L}}}^{T}}}{2}{\boldsymbol{ Q }^{T}} =
            {\boldsymbol{ Q }}{\boldsymbol{D }}{\boldsymbol{ Q }^{T}}

        所以，普通的增量形式本构方程 :math:`{\boldsymbol{\dot \sigma}}= \mathbb{C}:{{\boldsymbol{D}}}` 由于 :math:`\dot {\boldsymbol{\sigma}}` 不满足客观性，
        因此不再适用，需要做出相应的改变。主要体现为：

        选用客观的应力率取代普通的应力率，以保证客观性。本文采用 Cauchy 应力的 Zaremba-Jaumann 率 :math:`\hat{\boldsymbol{\sigma}}` (一般地简称为 Jaumann 率[1,2])。
        Jaumann 率是一种客观率(objective rate)，它使得在刚体转动中，在初始参考系下，初始的应力状态保持不变。

        下面详细介绍如何获得满足客观性要求的增量形式弹塑性本构关系。

        ----------------------------------------
        2. 运动学
        ----------------------------------------

        由于晶内部大量位错的存在，所以宏观上可以假设位错滑移在晶粒内部均匀分布。因而，在连续介质力学中，用变形梯度张量 :math:`{\boldsymbol{F}}`
        来描述滑移变形的宏观效应。 :math:`{\boldsymbol{F}}` 里包含了变形的所有信息，包括弹性变形和塑性变形，也包括了拉伸和旋转。
        采用 Hill 和 Rice 对晶体塑性变形几何学及运动学的理论表述方法，则晶体总的变形梯度 :math:`{\boldsymbol{F}}` ——
        当前构型量 :math:`{\boldsymbol{x}}` 对参考构型量 :math:`{\boldsymbol{X}}` 的偏导，可表示为：

        .. math::
            {\boldsymbol{F}} =\frac{\partial {\boldsymbol{x}}}{\partial {\boldsymbol{X}}} =
            {{\boldsymbol{F}}^{\text{e}}}{{\boldsymbol{F}}^{\text{p}}}

        其中， :math:`{{\boldsymbol{F}}^{\text{e}}}` 为晶格畸变和刚性转动产生的弹性变形梯度， :math:`{{\boldsymbol{F}}^{\text{p}}}`
        表示晶体沿着滑移方向的均匀剪切所产生的塑性变形梯度。

        图 1 晶体变形几何学示意图::



                                          最终构型
                                                           *
                                                     *      *
                                               *      *
                                         *      *      *     m^*
                                   *      *      *      *
                                    *      *      *             ^
                              n^*    *      *                     \
                               *      *                             \  F^e
                        ^       *                                     \
                       /                                                \
                      /                                                   \
                     /  F=F^eF^p                                            \
                    /                                                         \
               *  *  *  *  *                       F^p                            *  *  *  *  *
            n  *  *  *  *  *             ---------------------->            n     *  *  *  *  *
            ^  *  *  *  *  *                                                ^  *  *  *  *  *
            |  *  *  *  *  *                                                |  *  *  *  *  *
            ----->m                                                         ----->m
            初始构型                                                          中间构型


        上图所示为晶体变形几何学的示意图。可以看出，晶体滑移过程中，晶格矢量没有发生变化；但晶格的畸变会造成晶格矢量的变化，包括伸长和转动。

        用 :math:`{\boldsymbol{m}}^{(\alpha )}` 和 :math:`{\boldsymbol{n}}^{(\alpha )}` 分别表示变形前第 :math:`\alpha` 滑移系滑移方向和滑移面法向的单位向量。
        用 :math:`{\boldsymbol{m}}^{*\left( \alpha  \right)}` 和 :math:`{\boldsymbol{n}}^{*\left( \alpha \right)}` 分别表示晶格畸变后第 :math:`\alpha`
        滑移系的滑移方向和滑移面法向的单位向量。变形前与变形后第 :math:`\alpha` 滑移系的滑移方向和滑移面法向的单位向量存在下列关系：

        .. math::
            {{\boldsymbol{m}}^{*\left( \alpha  \right)}} = {{\boldsymbol{F}}^{\text{e}}}{{\boldsymbol{m}}^{\left( \alpha  \right)}},
            {{\boldsymbol{n}}^{*\left( \alpha  \right)}} = {{\boldsymbol{n}}^{\left( \alpha  \right)}} {\left( {{{\boldsymbol{F}}^{\rm{e}}}} \right)^{{\rm{ - }}1}}

        晶格畸变后，滑移面的滑移方向 :math:`{\boldsymbol{m}}^{*\left( \alpha  \right)}` 和法线方向 :math:`{\boldsymbol{n}}^{*\left( \alpha  \right)}` 不再是单位向量，
        但仍保持正交。

        自然的，可定义变形速度梯度，即变形速度 :math:`{\boldsymbol{v}}` 对当前构型 :math:`{\boldsymbol{x}}` 的导数，也被称为速度梯度张量 :math:`{\boldsymbol{L}}` ：

        .. math::
            {\boldsymbol{L}} = \frac{{\partial {\boldsymbol{v}}}}{{\partial {{\boldsymbol{x}}}}} = {\boldsymbol{\dot F}}{{\boldsymbol{F}}^{ - 1}}

        从引言的推导我们得知，这个速度梯度张量 :math:`{\boldsymbol{L}}` 不是客观张量，但由其得到的变形率张量 :math:`{\boldsymbol{D}}`
        和弹性变形率张量 :math:`{\boldsymbol{D}}^{\rm{e}}` 是客观张量。要得到弹性变形率张量 :math:`{\boldsymbol{D}}^{\rm{e}}`
        需要对 :math:`{\boldsymbol{L}}` 进行分解。

        对应于前述变形梯度的乘法分解，将速度梯度速度梯度张量 :math:`{\boldsymbol{L}}` 分解为与晶格畸变和刚体转动相对应的弹性部分 :math:`{{\boldsymbol{L}}^{\rm{e}}}`
        和与滑移相对应的塑性部分 :math:`{{\boldsymbol{L}}^{\rm{p}}}` ：

        .. math::
            {\boldsymbol{L}} = {{\boldsymbol{L}}^{\text{e}}}{{\boldsymbol{L}}^{\text{p}}}

        其中， :math:`{{\boldsymbol{L}}^{\rm{e}}}` 和 :math:`{{\boldsymbol{L}}^{\rm{p}}}` 分别为：

        .. math::
           {{\boldsymbol{L}}^{\rm{e}}} = {{{\boldsymbol{\dot F}}}^{\rm{e}}}{\left( {{{\boldsymbol{F}}^{\rm{e}}}} \right)^{ - 1}},

        .. math::
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

        速度梯度可以进一步分解为对称部分和反对称部分之和：

        .. math::
            {\boldsymbol{L}} = \frac{1}{2}\left( {{\boldsymbol{L}} + {{\boldsymbol{L}}^{\rm{T}}}} \right) + \frac{1}{2}\left( {{\boldsymbol{L}} - {{\boldsymbol{L}}^{\rm{T}}}} \right)

        将速度梯度的对称部分定义为变形率张量：

        .. math::
            {\boldsymbol{D}} = \frac{1}{2}\left( {{\boldsymbol{L}} + {{\boldsymbol{L}}^{\rm{T}}}} \right)={{\boldsymbol{D}}^{\rm{e}}}+{{\boldsymbol{D}}^{\rm{p}}}

        将速度梯度的反对称部分定义为旋率张量：

        .. math::
            {\boldsymbol{W}} = \frac{1}{2}\left( {{\boldsymbol{L}} - {{\boldsymbol{L}}^{\rm{T}}}} \right)={{\boldsymbol{W}}^{\rm{e}}}{\rm{ + }}{{\boldsymbol{W}}^{\rm{p}}}

        将变形率张量 :math:`{\boldsymbol{D}}` 分解得到弹性变形率张量 :math:`{{\boldsymbol{D}}^{\rm{e}}}` 与塑性变形率张量 :math:`{{\boldsymbol{D}}^{\rm{p}}}` ：

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

        以上晶体变形动力学的基本方程建立起了各滑移系剪切应变率与晶体宏观变形率之间的关系。下面将应力率、变形率及滑移剪切应变率联系起来，建立满足“客观性公理”的增量形式弹塑性本构关系。

        ----------------------------------------
        3. 弹塑性本构关系
        ----------------------------------------

        假设晶体材料的弹性性能不受滑移变形的影响，我们可建立满足“客观性公理”的增量形式弹性本构关系，表达式为：

        .. math::
            \hat{\boldsymbol{\sigma}}^{\rm{e}} = \mathbb{C}:{{\boldsymbol{D}}^{\rm{e}}}

        其中， :math:`{{\boldsymbol{D}}^{\rm{e}}}` 是变形率 :math:`{{\boldsymbol{D}}}` 的弹性部分，即弹性变形率张量。
        相应的， :math:`\hat{\boldsymbol{\sigma}}^{\rm{e}}` 是 Jaumann 率 :math:`\hat{\boldsymbol{\sigma}}` 的弹性部分，即弹性 Jaumann 率张量。

        以 图1 初始构形为基础的柯西应力张量的 Jaumann率 表达式为：

        .. math::
            {\boldsymbol {\hat \sigma }} = {\boldsymbol {\dot \sigma }} - {\boldsymbol{W}} \cdot {\boldsymbol{\sigma}} +
            {\boldsymbol{\sigma}} \cdot {\boldsymbol{W}}

        相应的将 :math:`\hat{\boldsymbol{\sigma}}^{\rm{e}}` 表示为以 图1 中间构形为基准状态的Kirchhoff应力张量的Jaumann导数，则有：

        .. math::
            \hat{\boldsymbol{\sigma}}^{\mathrm{e}} = {\boldsymbol {\dot \sigma }}  - {{\boldsymbol{W}}^{\rm{e}}} \cdot
            {\boldsymbol{\sigma}} + {\boldsymbol{\sigma}} \cdot {{\boldsymbol{W}}^{\rm{e}}}

        结合上述两式，将柯西应力率 :math:`{\boldsymbol {\dot \sigma }}` 作替换， :math:`\hat{\boldsymbol{\sigma}}` 可以表示为：

        .. math::
            \hat{\boldsymbol{\sigma}}  =\hat{\boldsymbol{\sigma}}^{\mathrm{e}}-\boldsymbol{W}^{\mathrm{p}} \cdot
            \boldsymbol{\sigma}+\boldsymbol{\sigma} \cdot \boldsymbol{W}^{\mathrm{p}}  ={\mathbb{C}}:{{\boldsymbol{D}}^{\rm{e}}}
            -{{\boldsymbol{W}}^{\rm{p}}} \cdot {\boldsymbol{\sigma }} + {\boldsymbol{\sigma }} \cdot {{\boldsymbol{W}}^{\rm{p}}}

        将 :math:`{{\boldsymbol{D}}^{\rm{e}}}` 和 :math:`{{\boldsymbol{W}}^{\rm{p}}}` 作替换，得到：

        .. math::
            {\boldsymbol{\hat \sigma }}  = \mathbb{C}:\left( {{\boldsymbol{D}} - {{\boldsymbol{D}}^{\rm{p}}}} \right) -
            \sum\limits_{\alpha  = 1}^N {{{\boldsymbol{Q}}^{\left( \alpha  \right)}}{{\dot \gamma }^{\left( \alpha  \right)}}}

        其中，

        .. math::
            {{\boldsymbol{Q}}^{\left( \alpha  \right)}} = {{\boldsymbol {\Omega }}^{\left( \alpha  \right)}} \cdot {\boldsymbol{\sigma}}
            - {\boldsymbol{\sigma}} \cdot {{\boldsymbol {\Omega }}^{\left( \alpha  \right)}}

        得到：

        .. math::
            {\boldsymbol{\hat \sigma }}= \mathbb{C}:\left( {{\boldsymbol{D}} - \sum\limits_{\alpha  = 1}^N {{{\boldsymbol{P}}^{\left(
            \alpha  \right)}}{{\dot \gamma }^{\left( \alpha  \right)}}} } \right) - \sum\limits_{\alpha  = 1}^N
            {{{\boldsymbol{Q}}^{\left( \alpha  \right)}}{{\dot \gamma }^{\left( \alpha  \right)}}}

        定义 :math:`\boldsymbol {S}^{(\alpha)}` 为：

        .. math::
            \boldsymbol {S}^{(\alpha)} = \mathbb{C}:{{\boldsymbol{P}}^{\left( \alpha  \right)}} +  {{\boldsymbol {\Omega }}^{\left( \alpha
            \right)}} \cdot {\boldsymbol{\sigma}} - {\boldsymbol{\sigma}} \cdot {{\boldsymbol {\Omega }}^{\left( \alpha  \right)}}

        最后得到Jaumann率表达式为：

        .. math::
            {\boldsymbol{\hat \sigma }}= \mathbb{C}:{\boldsymbol{D}} - \sum\limits_{\alpha  = 1}^N {\left[ {\mathbb{C}:
            {{\boldsymbol{P}}^{\left( \alpha  \right)}} +  {{\boldsymbol{\Omega }}^{\left( \alpha  \right)}} \cdot {\boldsymbol {\sigma }}
            - {\boldsymbol {\sigma }} \cdot {{\boldsymbol{\Omega }}^{\left( \alpha  \right)}}} \right]{{\dot \gamma }^{\left(
            \alpha  \right)}}}  = \mathbb{C}:{\boldsymbol{D}} - \sum\limits_{\alpha  = 1}^N { \boldsymbol {S}^{(\alpha)} {{\dot
            \gamma }^{\left( \alpha  \right)}}}

        上式将应力率、变形率及滑移剪切应变率联系起来，表示了应力率、变形率和滑移剪切应变率之间的定量关联，即增量形式的弹塑性本构关系。

        下一步的核心为确定所有可能开动滑移系的滑移剪切应变率 :math:`\dot{\gamma}^{(\alpha)}` 。

        ----------------------------------------
        4. 建立滑移系剪切应变演化唯象模型
        ----------------------------------------

        在晶体塑性本构模型中，需要通过各滑移系的剪切应变率计算应力率。因此，首先需要确定各滑移系剪切应变的演化方程。在剪切应变的硬化方程中，广泛地采用幂函数的形式，
        并且为了考虑晶体的循环塑性变形，引入各向同性强化项和随动强化项，建立如下混合强化模型：

        .. math::
            {{\dot \gamma }^{\left( \alpha  \right)}} =  {{\dot \gamma }_{0}^{\left( \alpha  \right)}}{f^{\left( \alpha  \right)}}
            \left( {\frac{{\left| {{\tau ^{\left( \alpha  \right)}} - {\alpha ^{\left( \alpha  \right)}}} \right| -
            {r^{\left( \alpha  \right)}}}}{{{K^{\left( \alpha  \right)}}}}} \right) = {{\dot \gamma }_{0}^{\left( \alpha  \right)}}
            {\rm{sign}}\left( {{\tau ^{\left( \alpha  \right)}} - {\alpha ^{\left( \alpha  \right)}}}
            \right){\left\langle {\frac{{\left| {{\tau ^{\left( \alpha  \right)}} - {\alpha ^{\left( \alpha  \right)}}}
            \right| - {r^{\left( \alpha  \right)}}}}{{{K^{\left( \alpha  \right)}}}}} \right\rangle ^{p_{\text{s}}}}

        式中， :math:`\dot{\gamma}^{(\alpha)}` 为滑移系 :math:`\alpha` 的剪切应变率； :math:`\tau^{(\alpha)}` 为滑移系 :math:`\alpha`
        的分解剪应力； :math:`\alpha^{(\alpha)}` 和 :math:`r^{(\alpha)}` 分别为滑移系 :math:`\alpha` 的背应力项（随动强化项）和各向同性强化项。
        另外， :math:`{{\dot \gamma }_{0}^{\left( \alpha  \right)}}` 为滑移系 :math:`\alpha` 的参考剪切应变率， :math:`{p_{\text{s}}}` 为应变速率敏感指数，
        当  :math:`{p_{\text{s}}} \rightarrow \infty` 时，接近于应变速率无关的情况，但此时计算不稳定。 :math:`K^{(\alpha)}` 为参考屈服强度，是取决于温度和滑移系种类的材料常数。

        下面介绍 :math:`\tau^{(\alpha)}` ， :math:`\alpha^{(\alpha)}` ， :math:`r^{(\alpha)}`
        的演化方程 :math:`{{\dot \tau }^{\left( \alpha  \right)}}` ， :math:`\dot{\alpha}^{(\alpha)}` 和 :math:`\dot{r}^{(\alpha)}` 。

        首先确定 :math:`{{\dot \tau }^{\left( \alpha  \right)}}` ，滑移系上的分解剪应力 :math:`{\tau ^{\left( \alpha  \right)}}` 定义为：

        .. math::
            {\tau ^{\left( \alpha  \right)}} = {{\boldsymbol{P}}^{\left( \alpha  \right)}}:{\boldsymbol{\sigma}}

        所以分解剪应力 :math:`{\tau ^{\left( \alpha  \right)}}` 对时间的导数 :math:`{\dot \tau }` 的表达形式为：

        .. math::
            {{\dot \tau }^{\left( \alpha  \right)}} = {{{\boldsymbol{\dot P}}}^{\left( \alpha  \right)}}:{\boldsymbol{\sigma}} +
            {{\boldsymbol{P}}^{\left( \alpha  \right)}}:{\boldsymbol{ \dot \sigma}}

        其中， :math:`{{{\boldsymbol{\dot P}}}^{\left( \alpha  \right)}}` 可用下列式子表示:

        .. math::
            {{{\boldsymbol{\dot P}}}^{\left( \alpha  \right)}} = {{\boldsymbol{D}}^{\rm{e}}}{{\boldsymbol{\Omega }}^{\left( \alpha  \right)}}
            + {{\boldsymbol{W}}^{\rm{e}}}{{\boldsymbol{P}}^{\left( \alpha  \right)}} - {{\boldsymbol{\Omega }}^{\left( \alpha  \right)}}{{\boldsymbol{D}}^{\rm{e}}}
            - {{\boldsymbol{P}}^{\left( \alpha  \right)}}{{\boldsymbol{W}}^{\rm{e}}}

        将 :math:`{{{\boldsymbol{\dot P}}}^{\left( \alpha  \right)}}` 和 :math:`{\boldsymbol{ \dot \sigma}}` 代入 :math:`{\dot \tau }`
        的表达式，并化简得到：

        .. math::
            {{\dot \tau }^{\left( \alpha  \right)}} =\left( {\mathbb{C}:{{\boldsymbol{P}}^{\left( \alpha
            \right)}} + {{\boldsymbol{\Omega }}^{\left( \alpha  \right)}}{\boldsymbol{\sigma }} - {\boldsymbol{\sigma }}
            {{\boldsymbol{\Omega }}^{\left( \alpha  \right)}}} \right):(\boldsymbol {D}-{{\boldsymbol{D}}^{\rm{p}}})

        将 :math:`\boldsymbol {S}^{(\alpha)}` 和 :math:`{{\boldsymbol{D}}^{\rm{p}}}` 代入上式得到 :math:`\tau^{(\alpha)}` 的演化方程：

        .. math::
            {{\dot \tau }^{\left( \alpha  \right)}}  = \boldsymbol {S}^{(\alpha)} :(\boldsymbol {D}-{{\boldsymbol{D}}^{\rm{p}}}) =
            \boldsymbol {S}^{(\alpha)} : \left( {\boldsymbol{D}}  - \sum\limits_{\alpha  = 1}^N {{{\boldsymbol{P}}^{\left( \alpha
            \right)}}{{\dot \gamma }^{\left( \alpha  \right)}}} \right)

        背应力 :math:`\dot{\alpha}^{(\alpha)}` 的演化方程为：

        .. math::
            \dot{\alpha}^{(\alpha)}=c_{1} \dot{\gamma}^{(\alpha)}-c_{2}\left|\dot{\gamma}^{(\alpha)}\right| \alpha^{(\alpha)}

        式中， :math:`c_{1} \dot{\gamma}^{(\alpha)}` 为背应力 :math:`\dot{\alpha}^{(\alpha)}` 关于剪切应变 :math:`\dot{\gamma}^{(\alpha)}`
        的线性项； :math:`c_{2}\left|\dot{\gamma}^{(\alpha)}\right| \alpha^{(\alpha)}` 为背应力 :math:`\dot{\alpha}^{(\alpha)}`
        关于剪切应变 :math:`\dot{\gamma}^{(\alpha)}` 的非线性项。 :math:`c_{1}` 和 :math:`c_{2}` 为取决于温度和滑移系种类的材料常数。

        其中各向同性强化项 :math:`\dot{r}^{(\alpha)}` 的演化方程为：

        .. math::
            {{\dot r}^{\left( \alpha  \right)}} = b_s Q_s \sum\limits_\beta  {{h_{\alpha \beta }}{{\dot \rho }^{\left( \beta  \right)}}}

        式中，引入各向同性状态变量  :math:`\rho^{(\beta)}` 描述晶体滑移中的位错硬化；引入交互作用系数矩阵  :math:`h_{\alpha \beta}` 描述滑移系间的交叉硬化，
        其对角线项表示滑移系的 “自硬化”，非对角线项表示由于滑移系间的耦合效应造成的 “潜在硬化”。对于给定的一组包含每个滑移系  :math:`\rho^{(\beta)}` 的位错状态，
        硬化将由系数 :math:`Q_s` 来确定。  :math:`b_s` 和  :math:`Q_s` 为取决于温度和滑移系种类的材料常数。因此有：

        .. math::
            r^{(\alpha)}=r_{0}^{(\alpha)}+b_s Q_s \sum_{\beta} h_{\alpha \beta} \rho^{(\beta)}

        滑移系 :math:`\alpha` 的各向同性强化项 :math:`r^{(\alpha)}` 由滑移系 :math:`\alpha` 的临界分解剪应力(初始阻应力) :math:`r_{0}^{(\alpha)}`
        和所有激活滑移系的位错状态变量 :math:`\rho^{(\beta)}` 共同确定。从冶金学的角度来看，
        该方程从左到右描述了基体带来的固溶硬化 :math:`r_{0}^{(\alpha)}` 和位错硬化 :math:`b_s Q_s \sum_{\beta} h_{\alpha \beta} \rho^{(\beta)}` 。

        在各向同性强化项 :math:`\dot{r}^{(\alpha)}` 中，采用非线性饱和形式的方程表示各向同性状态变量 :math:`\rho^{(\beta)}`，则有：

        .. math::
            \dot{\rho}^{(\beta)}=\left(1-b_s \rho^{(\beta)}\right)\left|\dot{\gamma}^{(\beta)}\right|

        其中，状态变量 :math:`\rho^{(\beta)}` 对应于可以进入 :math:`\gamma` 通道的临界位错密度。

        因此可以得到 :math:`\dot{r}^{(\alpha)}` 和 :math:`r^{(\alpha)}` 的完整表达式：

        .. math::
            \dot{r}^{(\alpha)}=b_s Q_s \sum_{\beta} h_{\alpha \beta}\left(1-b_s \rho^{(\beta)}\right)\left|\dot{\gamma}^{(\beta)}\right|

        .. math::
            r^{(\alpha)}=r_{0}^{(\alpha)}+b_s Q_s \sum_{\beta} h_{\alpha \beta}\left(1-b_s \rho^{(\beta)}\right)\left|\gamma^{(\beta)}\right|

        通过上述推导，方程 :math:`{{\dot \gamma }^{\left( \alpha  \right)}}` 建立了能够描述晶体循环变形中各向同性强化和随动强化的剪切应变硬化方程。
        利用计算得到的各滑移系中的剪切应变增量和晶体塑性理论中的本构关系，即可得到宏观应力增量，下面将详细介绍混合强化模型的数值离散过程。

        ----------------------------------------
        5 数值离散求解
        ----------------------------------------

        如果将晶体塑性本构模型与基于位移场的求解有限元软件相结合，主要包含两个基本任务：一是通过积分点处的变形计算应力值，二是更新当前积分点的一致性切线刚度矩阵。
        而应力和切线刚度矩阵的更新则依赖于所有开动滑移系的剪切应变增量 :math:`\Delta \gamma^{(\alpha)}` 的求解。

        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        5.1 求解应力应变增量以及相关内变量初值
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

        下面，确定 :math:`\tau^{(\alpha)}` ， :math:`\alpha^{(\alpha)}` ， :math:`r^{(\alpha)}`
        的离散形式 :math:`\Delta \tau^{(\alpha)}` ， :math:`\Delta \alpha^{(\alpha)}` ， :math:`\Delta r^{(\alpha)}` 。

        对 :math:`{\dot \tau }` 进行积分，得到分解剪应力离散格式 :math:`\Delta \tau^{(\alpha)}` ：

        .. math::
            \Delta \tau^{(\alpha)} = {{\dot \tau }^{\left( \alpha  \right)}} \Delta t=\boldsymbol{S}^{(\alpha)}: \Delta
            \boldsymbol{\varepsilon}-\boldsymbol {S}^{(\alpha)}: \sum_{\beta=1}^{N} \boldsymbol {P}^{(\beta)} \Delta \gamma^{(\beta)}
            =  \boldsymbol{S}^{(\alpha)}: \left(\Delta \boldsymbol{\varepsilon}- \sum_{\beta=1}^{N} \boldsymbol {P}^{(\beta)} \Delta
            \gamma^{(\beta)} \right)

        对 :math:`{\dot \alpha }` 进行积分，得到背应力的离散格式 :math:`\Delta \alpha^{(\alpha)}` ：

        .. math::
            \Delta \alpha^{(\alpha)}=c_{1} \Delta \gamma^{(\alpha)}-c_{2}\left|\Delta \gamma^{(\alpha)}\right| \alpha^{(\alpha)}

        对 :math:`{\dot r }` 进行积分，得到各向同性强化项的离散格式 :math:`\Delta r^{(\alpha)}` ：

        .. math::
            \Delta r^{(\alpha)}=b_s Q_s \sum_{\beta} h_{\alpha \beta}\left(1-b_s \rho^{(\beta)}\right)\left|\Delta \gamma^{(\beta)}\right|

        令：

        .. math::
            \left\langle X^{\left( \alpha  \right)} \right\rangle  = \left\langle {\frac{{\left| {{\tau ^{\left( \alpha  \right)}} - {\alpha
            ^{\left( \alpha  \right)}}} \right| - {r^{\left( \alpha  \right)}}}}{{{K^{\left( \alpha  \right)}}}}} \right\rangle

        得到混合强化模型：

        .. math::
            {f^{\left( \alpha  \right)}} = {\rm{sign}}\left( {{\tau ^{\left( \alpha  \right)}} - {\alpha ^{(\alpha )}}}
            \right){\left\langle X^{\left( \alpha  \right)} \right\rangle ^{p_s}}

        接下来采用链式法则，对每一项分别求导：

        .. math::
            \frac{{\partial {f^{\left( \alpha  \right)}}}}{{\partial \left\langle X^{\left( \alpha  \right)} \right\rangle }} = {\rm{sign}}\left(
            {{\tau ^{\left( \alpha  \right)}} - {\alpha ^{(\alpha )}}} \right){p_{\text{s}}}{\left\langle X^{\left( \alpha  \right)} \right\rangle ^{{p_s}{\rm{ - 1}}}}

        .. math::
            \frac{{\partial \left\langle X^{\left( \alpha  \right)} \right\rangle }}{{\partial {\tau ^{\left( \alpha  \right)}}}} = \frac{1}{{{K^{
            \left( \alpha  \right)}}}}{\rm{sign}}\left( {{\tau ^{\left( \alpha  \right)}} - {\alpha ^{\left( \alpha  \right)}}} \right) H(X^{\left( \alpha  \right)})

        .. math::
            \frac{{\partial \left\langle X^{\left( \alpha  \right)} \right\rangle }}{{\partial {\alpha ^{\left( \alpha  \right)}}}} =  - \frac{1}{{{K^{\left( \alpha
             \right)}}}}{\rm{sign}}\left( {{\tau ^{\left( \alpha  \right)}} - {\alpha ^{\left( \alpha  \right)}}} \right) H(X^{\left( \alpha  \right)})

        .. math::
            \frac{{\partial \left\langle X^{\left( \alpha  \right)} \right\rangle }}{{\partial {r^{\left( \alpha  \right)}}}} =
            - \frac{1}{{{K^{\left( \alpha  \right)}}}} H(X^{\left( \alpha  \right)})

        所以有：

        .. math::
            \frac{{\partial {{\dot \gamma }^{\left( \alpha  \right)}}}}{{\partial {\tau ^{\left( \alpha  \right)}}}} =
            {{\dot \gamma }_{0}^{\left( \alpha  \right)}}\frac{{\partial {f^{\left( \alpha  \right)}}}}{{\partial {\tau ^{\left(
            \alpha  \right)}}}}  = {{\dot \gamma }_{0}^{\left( \alpha  \right)}}\frac{{\partial {f^{\left( \alpha  \right)}}}}
            {{\partial \left| X^{\left( \alpha  \right)} \right|}}\frac{{\partial \left| X^{\left( \alpha  \right)} \right|}}{{\partial {\tau ^{\left( \alpha  \right)}}}}
            = {{\dot \gamma }_{0}^{\left( \alpha  \right)}}{p_{\text{s}}}{\left\langle X^{\left( \alpha  \right)} \right\rangle ^{{p_s}{\rm{ - 1}}}}\frac{1}{{{K^{\left(
            \alpha  \right)}}}} H(X^{\left( \alpha  \right)})

        .. math::
            \frac{{\partial {{\dot \gamma }^{\left( \alpha  \right)}}}}{{\partial {\alpha ^{\left( \alpha  \right)}}}} =
            {{\dot \gamma }_{0}^{\left( \alpha  \right)}}\frac{{\partial {f^{\left( \alpha  \right)}}}}{{\partial {\alpha ^{\left(
            \alpha  \right)}}}}  = {{\dot \gamma }_{0}^{\left( \alpha  \right)}}\frac{{\partial {f^{\left( \alpha  \right)}}}}
            {{\partial \left| X^{\left( \alpha  \right)} \right|}}\frac{{\partial \left| X^{\left( \alpha  \right)} \right|}}{{\partial {\alpha ^{\left( \alpha  \right)}}}}
            = - {{\dot \gamma }_{0}^{\left( \alpha  \right)}}{p_{\text{s}}}{\left\langle X^{\left( \alpha  \right)} \right\rangle ^{{p_s}{\rm{ - 1}}}}\frac{1}{{{K^{\left(
            \alpha  \right)}}}} H(X^{\left( \alpha  \right)})

        .. math::
            \frac{{\partial {{\dot \gamma }^{\left( \alpha  \right)}}}}{{\partial {r^{\left( \alpha  \right)}}}} =
            {{\dot \gamma }_{0}^{\left( \alpha  \right)}}\frac{{\partial {f^{\left( \alpha  \right)}}}}{{\partial {r^{\left(
            \alpha  \right)}}}} = {{\dot \gamma }_{0}^{\left( \alpha  \right)}}\frac{{\partial {f^{\left( \alpha
            \right)}}}}{{\partial \left| X^{\left( \alpha  \right)} \right|}}\frac{{\partial \left| X^{\left( \alpha  \right)} \right|}}{{\partial {r^{\left( \alpha  \right)}}}} =
            - {{\dot \gamma }_{0}^{\left( \alpha  \right)}}{p_{\text{s}}}{\left\langle X^{\left( \alpha  \right)} \right\rangle ^{{p_s}{\rm{ - 1}}}}\frac{1}{{{K^{\left(
            \alpha  \right)}}}}{\rm{sign}}\left( {{\tau ^{\left( \alpha  \right)}} - {\alpha ^{(\alpha )}}} \right) H(X^{\left( \alpha  \right)})

        将上面的式子带入式 :math:`\Delta \gamma^{(\alpha)}` ，将每个滑移系 :math:`\alpha` 的剪切应变增量 :math:`\Delta \gamma^{(\alpha)}` 写成矩阵形式，可得：

        .. math::
            \Delta {\gamma ^{\left( \alpha  \right)}} = \Delta t\left[ \begin{array}{l}
            {{\dot \gamma }^{\left( \alpha  \right)}}\left( t \right)\\
             + \theta {{\dot \gamma }_{0}^{\left( \alpha  \right)}}{p_{\text{s}}}{\left\langle X^{\left( \alpha  \right)} \right\rangle ^{{p_s}{\rm{ - 1}}}}\frac{1}{{{K^{\left( \alpha  \right)}}}} H(X^{\left( \alpha  \right)}) \Delta {\tau ^{\left( \alpha  \right)}} \\
             - \theta {{\dot \gamma }_{0}^{\left( \alpha  \right)}}{p_{\text{s}}}{\left\langle X^{\left( \alpha  \right)} \right\rangle ^{{p_s}{\rm{ - 1}}}}\frac{1}{{{K^{\left( \alpha  \right)}}}} H(X^{\left( \alpha  \right)}) \Delta {\alpha ^{\left( \alpha  \right)}}\\
             - \theta {{\dot \gamma }_{0}^{\left( \alpha  \right)}}{p_{\text{s}}}{\left\langle X^{\left( \alpha  \right)} \right\rangle ^{{p_s}{\rm{ - 1}}}}\frac{1}{{{K^{\left( \alpha  \right)}}}} H(X^{\left( \alpha  \right)}) {\rm{sign}}\left( {{\tau ^{\left( \alpha  \right)}} - {\alpha ^{(\alpha )}}} \right)\Delta {r^{\left( \alpha  \right)}}
            \end{array} \right]

        进一步将 :math:`\Delta \tau^{(\alpha)}` ， :math:`\Delta \alpha^{(\alpha)}` 和 :math:`\Delta r^{(\alpha)}`
        的表达式代入剪切应变增量方程 :math:`\Delta \gamma^{(\alpha)}` ，同时注意到 :math:`\langle X^{\left( \alpha  \right)}\rangle^{{p_s}-1} H(X^{\left( \alpha  \right)})=\langle X^{\left( \alpha  \right)}\rangle^{{p_s}-1}` ，可以得到：

        .. math::
            \sum\limits_{\beta  = 1}^N {{\delta _{\alpha \beta }}\Delta {\gamma ^{\left( \beta  \right)}}} {\rm{ = }}\Delta t\left[ \begin{array}{l}
            {{\dot \gamma }^{\left( \alpha  \right)}}\left( t \right)\\
             + \theta {{\dot \gamma }_{0}^{\left( \alpha  \right)}}{p_{\text{s}}}{\left\langle X^{\left( \alpha  \right)} \right\rangle ^{{p_s}{\rm{ - 1}}}}\frac{1}{{{K^{\left( \alpha  \right)}}}}\left[ {{{\boldsymbol{S}}^{\left( \alpha  \right)}}:\Delta {\boldsymbol{\varepsilon }} - {{\boldsymbol{S}}^{\left( \alpha  \right)}}:\sum\limits_{\beta  = 1}^N {{{\boldsymbol{P}}^{\left( \beta  \right)}}\Delta {\gamma ^{\left( \beta  \right)}}} } \right]\\
             - \theta {{\dot \gamma }_{0}^{\left( \alpha  \right)}}{p_{\text{s}}}{\left\langle X^{\left( \alpha  \right)} \right\rangle ^{{p_s}{\rm{ - 1}}}}\frac{1}{{{K^{\left( \alpha  \right)}}}}\left( {{c_1}\sum\limits_{\beta  = 1}^N {{\delta _{\alpha \beta }}\Delta {\gamma ^{\left( \beta  \right)}}}  - {c_2}{\rm{sign}}\left( {{{\dot \gamma }^{\left( \beta  \right)}}} \right)\sum\limits_{\beta  = 1}^N {{\delta _{\alpha \beta }}\Delta {\gamma ^{\left( \beta  \right)}}} {\alpha ^{\left( \alpha  \right)}}} \right)\\
             - \theta {{\dot \gamma }_{0}^{\left( \alpha  \right)}}{p_{\text{s}}}{\left\langle X^{\left( \alpha  \right)} \right\rangle ^{{p_s}{\rm{ - 1}}}}\frac{1}{{{K^{\left( \alpha  \right)}}}}{\rm{sign}}\left( {{\tau ^{\left( \alpha  \right)}} - {\alpha ^{(\alpha )}}} \right)bQ\sum\limits_\beta  {{h_{\alpha \beta }}\left( {1 - b_s{\rho ^{\left( \beta  \right)}}} \right)\left| {\Delta {\gamma ^{\left( \beta  \right)}}} \right|}
            \end{array} \right]

        合并式子：

        .. math::
            \begin{aligned}
            \begin{array}{l}
            \sum\limits_{\beta  = 1}^N {\left[ \begin{array}{l}
            {\delta _{\alpha \beta }}\\
             + \Delta t\theta {{\dot \gamma }_{0}^{\left( \alpha  \right)}}{p_{\text{s}}}{\left\langle X^{\left( \alpha  \right)} \right\rangle ^{{p_s}{\rm{ - 1}}}}\frac{1}{{{K^{\left( \alpha  \right)}}}}{{\boldsymbol{S}}^{\left( \alpha  \right)}}:{{\boldsymbol{P}}^{\left( \beta  \right)}}\\
             + \Delta t\theta {{\dot \gamma }_{0}^{\left( \alpha  \right)}}{p_{\text{s}}}{\left\langle X^{\left( \alpha  \right)} \right\rangle ^{{p_s}{\rm{ - 1}}}}\frac{1}{{{K^{\left( \alpha  \right)}}}}{\delta _{\alpha \beta }}\left( {{c_1} - {c_2}{\rm{sign}}\left( {{{\dot \gamma }^{\left( \beta  \right)}}} \right){\alpha ^{\left( \alpha  \right)}}} \right)\\
             + \Delta t\theta {{\dot \gamma }_{0}^{\left( \alpha  \right)}}{p_{\text{s}}}{\left\langle X^{\left( \alpha  \right)} \right\rangle ^{{p_s}{\rm{ - 1}}}}\frac{1}{{{K^{\left( \alpha  \right)}}}}{\rm{sign}}\left( {{\tau ^{\left( \alpha  \right)}} - {\alpha ^{(\alpha )}}} \right)bQ{h_{\alpha \beta }}\left( {1 - b_s{\rho ^{\left( \beta  \right)}}} \right){\rm{sign}}\left( {{{\dot \gamma }^{\left( \beta  \right)}}} \right)
            \end{array} \right]\Delta {\gamma ^{\left( \beta  \right)}}} \\
             = \Delta t{{\dot \gamma }^{\left( \alpha  \right)}}\left( t \right) + \Delta t\theta {{\dot \gamma }_{0}^{\left( \alpha  \right)}}{p_{\text{s}}}{\left\langle X^{\left( \alpha  \right)} \right\rangle ^{{p_s}{\rm{ - 1}}}}\frac{1}{{{K^{\left( \alpha  \right)}}}}{{\boldsymbol{S}}^{\left( \alpha  \right)}}:\Delta {\boldsymbol{\varepsilon }}
            \end{array}
            \end{aligned}

        编程过程中为了简化计算，引入一些临时变量：

        .. math::
            term1 := \Delta t \theta

        .. math::
            term2 := term1 \cdot {{\dot \gamma }_{0}^{\left( \alpha  \right)}} {p_{\text{s}}}\langle X^{\left( \alpha  \right)}\rangle^{{p_s}-1} \cdot \frac{1}{{{K^{\left( \alpha  \right)}}}}

        .. math::
            term4 := \boldsymbol{S}^{(\alpha)}: \boldsymbol{P}^{(\beta)}

        整理得到引入线性背应力项（随动强化项）和各向同性强化模型的迭代格式为由 :math:`N` 个末知数 :math:`\Delta \gamma^{(\beta)}` 和 :math:`N` 个非线性方程组成的方程组：

        .. math::
            \begin{aligned}
            & \left[\begin{array}{l}
            \delta_{\alpha \beta} \\
            +term2 \cdot term4\\
            + term2 \cdot {\delta _{\alpha \beta }}\left( {{c_1} - {c_2}{\rm{sign}}\left( {{{\dot \gamma }^{\left( \beta  \right)}}} \right){\alpha ^{\left( \alpha  \right)}}} \right)\\
            {\rm{ + }}term2 \cdot {\rm{sign}}\left( {{\tau ^{\left( \alpha  \right)}} - {\alpha ^{(\alpha )}}} \right) \cdot bQ{h_{\alpha \beta }}\left( {1 - b_s{\rho ^{\left( \beta  \right)}}} \right) \cdot {\rm{sign}}\left( {{{\dot \gamma }^{\left( \beta  \right)}}} \right)
            \end{array}\right] \Delta \gamma^{(\alpha)} \\
            & =\Delta t \dot{\gamma}^{(\alpha)}(t)+term2 \cdot \boldsymbol {S}^{(\alpha)}: \Delta {\boldsymbol{\varepsilon }}
            \end{aligned}

        其中，上标 :math:`\alpha` 表示第 :math:`\alpha` 个滑移系( :math:`\alpha=1 \sim N，N` 为所有可能开动滑移系的数目)，
        等式右边的 :math:`\Delta {\boldsymbol{\varepsilon }}` 为已知项。求解该非线性方程组可以得到所有滑移系的初始剪切应变增量 :math:`\Delta \gamma^{(\alpha)}`，
        进而计算应力增量 :math:`\Delta \boldsymbol{\sigma}` 和其他状态变量的初始增量。

        将上面的方程组简写为以下形式：

        .. math::
            \boldsymbol {A} \Delta {\gamma ^{\left( \alpha  \right)}} = \Delta t{{\dot \gamma }^{\left( \alpha  \right)}}\left( t \right)
            + term2 \cdot {{\boldsymbol{S}}^{\left( \alpha  \right)}}:\Delta {\boldsymbol{\varepsilon }}

        方程两边对 :math:`\Delta {\boldsymbol{\varepsilon }}` 求偏导得到：

        .. math::
            \boldsymbol {A} \frac{{\partial \Delta {\gamma ^{\left( \alpha  \right)}}}}{{\partial \Delta {\boldsymbol{\varepsilon }}}} =
            term2 \cdot {{\boldsymbol{S}}^{\left( \alpha  \right)}}

        所以有：

        .. math::
            ddgdde^{(\alpha)} = \frac{{\partial \Delta {\gamma ^{\left( \alpha  \right)}}}}{{\partial \Delta {\boldsymbol{\varepsilon }}}} =
            {{\boldsymbol {A}}^{ - 1}} \cdot term2 \cdot {{\boldsymbol{S}}^{\left( \alpha  \right)}}

        在第三节，我们已经得到 Jaumann率 表达式为：

        .. math::
            {\boldsymbol{\hat \sigma }} = \mathbb{C}:{\boldsymbol{D}} - \sum\limits_{\alpha  = 1}^N { \boldsymbol {S}^{(\alpha)} {{\dot
            \gamma }^{\left( \alpha  \right)}}}

        对 Jaumann 率进行时间积分，我们得到：

        .. math::
            \Delta {\boldsymbol{\sigma}} = \mathbb{C}:{\Delta {\boldsymbol{\varepsilon }}} - \sum\limits_{\alpha  = 1}^N
            {\boldsymbol {S}^{(\alpha)} {\Delta { \gamma }^{\left( \alpha  \right)}}}

        进而，我们可以得到，弹性模量张量的弹性部分， 即  :math:`ddsdde` 矩阵为：

        .. math::
            ddsdde = \frac{\partial {\Delta {\boldsymbol{\sigma }^{(\alpha)}}}}{\partial {\Delta \boldsymbol{\varepsilon }}} =
            \mathbb{C} - \sum\limits_{\alpha = 1}^N { \boldsymbol {S}^{(\alpha)} \frac{{\partial \Delta {\gamma ^{\left( \alpha  \right)}}}}{{\partial \Delta {\boldsymbol{\varepsilon }}}}}
             = \mathbb{C} - \sum\limits_{\alpha = 1}^N \boldsymbol {S}^{(\alpha)} \cdot ddgdde^{(\alpha)}


        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        5.2 迭代求解剪切应变增量以及更新切线刚度矩阵
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        采用牛顿拉夫森迭代方法进行迭代求解。在上面的推导中，初始剪切应变增量 :math:`{{{\left\{ {\Delta {\gamma ^{\left( \alpha  \right)}}} \right\}}^{\left( k \right)}}}`
        是我们利用 5.1 节的非线性方程组求出的近似值。假设初始剪切应变增量与真实剪切应变增量 :math:`\Delta \gamma^{(\alpha)}` 之间的误差 Residual为 :math:`R` 。

        我们可以写出 :math:`R` 的表达式：

        .. math::
            R = F \left( {{{\left\{ {\Delta {\gamma ^{\left( \alpha  \right)}}} \right\}}^{\left( k \right)}}} \right)
            = {{{\left\{ {\Delta {\gamma ^{\left( \alpha  \right)}}} \right\}}^{\left( k \right)}}} - \Delta t\left( {1 - \theta } \right){{\dot \gamma }^{\left( \alpha  \right)}}\left( t \right) -
            \Delta t\theta {{\dot \gamma }^{\left( \alpha  \right)}}\left( {t + \Delta t} \right)

        上式就是牛顿拉弗森法迭代的目标函数。我们要做的就是对这个函数上的点做切线，并求切线的零点。即使得Residual为 0 或接近我们的预设阈值 tolerance ，可用数学式表达为：

        .. math::
            F'\left( {{{\left\{ {\Delta {\gamma ^{\left( \alpha  \right)}}} \right\}}^{\left( k \right)}}} \right)
            \cdot \Delta {\left\{ {\Delta {\gamma ^{\left( \alpha  \right)}}} \right\}^{\left( {k + 1} \right)}} =
            0 - F\left( {{{\left\{ {\Delta {\gamma ^{\left( \alpha  \right)}}} \right\}}^{\left( k \right)}}} \right)

        其中，

        .. math::
            {\left\{ {\Delta {\gamma ^{\left( \alpha  \right)}}} \right\}^{\left( {k + 1} \right)}} =
            {\left\{ {\Delta {\gamma ^{\left( \alpha  \right)}}} \right\}^{\left( k \right)}} +
            \Delta {\left\{ {\Delta {\gamma ^{\left( \alpha  \right)}}} \right\}^{\left( {k + 1} \right)}}

        当初值计算完成后，获得了新的 :math:`\left\{ {X^{\left( \alpha  \right)}} \right\}^{\left( {k + 1} \right)}` ：

        .. math::
            \left\{ {X^{\left( \alpha  \right)}} \right\}^{\left( {k + 1} \right)} = \frac{{\left| {\tau _t^{\left( \alpha  \right)} + {\left\{ {\Delta {\tau ^{\left( \alpha  \right)}}} \right\}^{\left( {k + 1} \right)}} - \left(
            {\alpha _t^{\left( \alpha  \right)} +{\left\{ {\Delta {\alpha ^{\left( \alpha  \right)}}} \right\}^{\left( {k + 1} \right)}}} \right)} \right| -
            \left( {r_t^{\left( \alpha  \right)} + {\left\{ {\Delta {r^{\left( \alpha  \right)}}} \right\}^{\left( {k + 1} \right)}}} \right)}}{{{K^{\left( \alpha  \right)}}}}

        进而可得到即将用于牛顿拉夫森迭代的剪切应变速率表达式：

        .. math::
            {\left\{ {\dot \gamma _{t + \Delta t}^{\left( \alpha  \right)}} \right\}^{\left( {k + 1} \right)}} = {{\dot \gamma }_{0}^{\left( \alpha  \right)}}{\left\langle \left\{ {X^{\left( \alpha  \right)}} \right\}^{\left( {k + 1} \right)}
            \right\rangle ^{p_s}}{\text{sign}}\left( {\tau _t^{\left( \alpha  \right)} + {\left\{ {\Delta {\alpha ^{\left( \alpha  \right)}}} \right\}^{\left( {k + 1} \right)}} - \left(
            {\alpha _t^{\left( \alpha  \right)} +{\left\{ {\Delta {\alpha ^{\left( \alpha  \right)}}} \right\}^{\left( {k + 1} \right)}}} \right)} \right)

        参考 5.1 节的推导，最后我们得到迭代求解该非线性方程组的所有滑移系的剪切应变增量的方程组：

        .. math::
            {\boldsymbol{A}}^{\left( k + 1 \right)} \left( {{{\left\{ {\Delta {\gamma ^{\left( \alpha  \right)}}} \right\}}^{\left( k \right)}}} \right)
            \cdot \Delta {\left\{ {\Delta {\gamma ^{\left( \alpha  \right)}}} \right\}^{\left( {k + 1} \right)}} = \left\{ {rhs} \right\}^{\left( {k} \right)}

        其中，刚度矩阵 :math:`{\boldsymbol{A}}^{\left( k + 1 \right)} \left( {{{\left\{ {\Delta {\gamma ^{\left( \alpha  \right)}}} \right\}}^{\left( k \right)}}} \right)` 为：

        .. math::
           {\boldsymbol{A}}^{\left( k + 1 \right)}  =
            F'\left( {{{\left\{ {\Delta {\gamma ^{\left( \alpha  \right)}}} \right\}}^{\left( k \right)}}} \right) = \left[ {\begin{array}{*{20}{l}}
              {{\delta _{\alpha \beta }}} \\
              { + term2 \cdot term4} \\
              { + term2 \cdot {\delta _{\alpha \beta }}\left( {{c_1} - {c_2}{\alpha ^{(\alpha )}}{\text{sign}}\left( {{{\left\{ {\Delta {\gamma ^{\left( \alpha  \right)}}} \right\}}^{\left( k \right)}}} \right)} \right)} \\
              { + term2 \cdot bQ{h_{\alpha \beta }}\left( {1 - b_s{\rho ^{(\beta )}}} \right){\delta _{\alpha \beta }}{\text{sign}}\left( {{{\left\{ {\Delta {\gamma ^{\left( \alpha  \right)}}} \right\}}^{\left( k \right)}}} \right){\text{sign}}\left( {\tau _t^{\left( \alpha  \right)} + {{\left\{ {\Delta {\tau ^{\left( \alpha  \right)}}} \right\}}^{\left( {k + 1} \right)}} - \left( {\alpha _t^{\left( \alpha  \right)} + {{\left\{ {\Delta {\alpha ^{\left( \alpha  \right)}}} \right\}}^{\left( {k + 1} \right)}}} \right)} \right)}
            \end{array}} \right]

        可以看出，用于迭代的刚度矩阵 :math:`{\boldsymbol{A}}^{\left( k + 1 \right)}` 与求解初值的刚度矩阵 :math:`{\boldsymbol{A}}` 形式一致。其中，

        .. math::
            term1 := \Delta t \theta

        .. math::
            term2 := term1 \cdot {{\dot \gamma }_{0}^{\left( \alpha  \right)}} {p_{\text{s}}}\langle \left\{ {X^{\left( \alpha  \right)}} \right\}^{\left( {k + 1} \right)} \rangle^{{p_s}-1} \cdot \frac{1}{{{K^{\left( \alpha  \right)}}}}

        .. math::
            term4 := \boldsymbol{S}^{(\alpha)}: \boldsymbol{P}^{(\beta)}

        方程组右边项 :math:`rhs` 为：

        .. math::
            \left\{ {rhs} \right\}^{\left( {k} \right)} =  - F\left( {{{\left\{ {\Delta {\gamma ^{\left( \alpha  \right)}}} \right\}}^{\left( k \right)}}} \right)
            = \Delta t\left( {1 - \theta } \right){{\dot \gamma }^{\left( \alpha  \right)}}\left( t \right) +
            \Delta t\theta {{\dot \gamma }^{\left( \alpha  \right)}}\left( {t + \Delta t} \right) - {{{\left\{ {\Delta {\gamma ^{\left( \alpha  \right)}}} \right\}}^{\left( k \right)}}}

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
        {\boldsymbol{\sigma}} \cdot {{\rm{\Omega }}^{\left( \alpha  \right)}}` ：Q

        Jaumann率中的旋转部分 :math:`\mathbb{C}:{{\boldsymbol{P}}^{\left( \alpha  \right)}} +  {{\boldsymbol {\Omega }}^{\left(
        \alpha \right)}} \cdot {\boldsymbol{\sigma}} - {\boldsymbol{\sigma}} \cdot {{\rm{\Omega }}^{\left( \alpha  \right)}}` ：S

        弹性模量张量 :math:`\mathbb{C}` ：C

        剪切应变 :math:`\gamma` ：gamma

        滑移系的剪切应变率 :math:`\dot{\gamma}` ：gamma_dot

        滑移系的应变速率敏感指数 :math:`{p_{\text{s}}}` ：p_s

        滑移系的参考剪切应变率 :math:`\dot{a}` ：dot_gamma_0

        参考屈服强度 :math:`K^{(\alpha)}` ：K

        剪切应变速率初值 :math:`\Delta \gamma_{t}^{(\alpha)}` ：gamma_dot_t

        用于迭代的剪切应变速率 :math:`\Delta \gamma_{t+\Delta t}^{(\alpha)}` ：gamma_dot

        混合强化模型的中间变量 :math:`X^{\left( \alpha  \right)}` ：X

        背应力项参数 :math:`c_{1}` ：c_1

        背应力项参数 :math:`c_{2}` ：c_2

        各向同性强化项的临界分解剪应力(初始阻应力) :math:`r_{0}` ：r_0

        各向同性强化项参数 :math:`b_s` ：b_s

        各向同性强化项参数 :math:`Q`  ：Q_s

        参考文献：

        [1] 近代连续介质力学. 赵亚溥.

        [2] Nonlinear Finite Elements for Continua and Structures, Ted Belytschko.
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
        b_s = self.b_s
        Q_s = self.Q_s
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

            # Q = Omega * stress - stress * Omega
            # S = dot(P, C) + Q
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
            A += term2 * b_s * Q_s * H * (1.0 - b_s * rho) * sign(tau - alpha) * sign(gamma_dot)

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
            delta_rho = (1.0 - b_s * rho) * abs(delta_gamma)
            delta_r = b_s * Q_s * dot(H, delta_rho)

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

        strain_energy = 0.5 * sum(strain * stress)

        if self.section.type == 'PlaneStrain':
            ddsdde = delete(delete(ddsdde, [2, 4, 5], axis=0), [2, 4, 5], axis=1)
            stress = delete(stress, [2, 4, 5])

        output = {'stress': stress, 'strain_energy': strain_energy}

        return ddsdde, output


if __name__ == "__main__":
    from pyfem.utils.visualization import print_slots_dict

    print_slots_dict(PlasticCrystal.__slots_dict__)

    from pyfem.Job import Job

    # job = Job(r'..\..\..\examples\mechanical\1element\hex20_crystal\Job-1.toml')
    job = Job(r'..\..\..\examples\mechanical\4_grains_crystal\Job-1.toml')

    job.run()