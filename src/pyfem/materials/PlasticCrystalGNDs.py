# -*- coding: utf-8 -*-
"""

"""
from copy import deepcopy

import numpy as np

from pyfem.fem.Timer import Timer
from pyfem.fem.constants import DTYPE
from pyfem.io.Material import Material
from pyfem.io.Section import Section
from pyfem.materials.BaseMaterial import BaseMaterial
from pyfem.materials.crystal_slip_system import generate_mn
from pyfem.utils.colors import error_style
from pyfem.utils.mechanics import get_transformation, get_voigt_transformation


class PlasticCrystalGNDs(BaseMaterial):
    r"""
    晶体塑性材料。

    支持的截面属性：('Volume', 'PlaneStrain')

    :ivar tolerance: 误差容限
    :vartype tolerance: float

    :ivar total_number_of_slips: 总滑移系数量 [-]
    :vartype total_number_of_slips: int

    :ivar elastic: 弹性参数字典
    :vartype elastic: dict

    :ivar G: 剪切模量 [Pa]
    :vartype G: float

    :ivar k_b: 玻尔兹曼常数 [J/K]
    :vartype k_b: float

    :ivar temperature: 温度 [K]
    :vartype temperature: float

    :ivar C: 弹性矩阵 [Pa]
    :vartype C: np.ndarray

    :ivar slip_system_name: 滑移系统名称
    :vartype slip_system_name: list[str]

    :ivar c_over_a: 晶体坐标系的c/a [-]
    :vartype c_over_a: list[float]

    :ivar theta: 切线系数法参数
    :vartype theta: float

    :ivar H: 硬化系数矩阵 [-]
    :vartype H: np.ndarray

    :ivar tau_sol: 固溶强度 [Pa]
    :vartype tau_sol: np.ndarray

    :ivar v_0: 位错滑移参考速度 [m/s]
    :vartype v_0: np.ndarray

    :ivar b_s: 位错滑移柏氏矢量长度 [m]
    :vartype b_s: np.ndarray

    :ivar Q_s: 位错滑移激活能 [J]
    :vartype Q_s: np.ndarray

    :ivar p_s: 位错滑移阻力拟合参数 [-]
    :vartype p_s: np.ndarray

    :ivar q_s: 位错滑移阻力拟合参数 [-]
    :vartype q_s: np.ndarray

    :ivar d_grain: 平均晶粒尺寸 [m]
    :vartype d_grain: np.ndarray

    :ivar i_slip: 平均位错间隔拟合参数 [-]
    :vartype i_slip: np.ndarray

    :ivar c_anni: 位错湮灭拟合参数 [-]
    :vartype c_anni: np.ndarray

    :ivar Q_climb: 位错攀移激活能 [J]
    :vartype Q_climb: np.ndarray

    :ivar D_0: 自扩散系数因子 [m^2/s]
    :vartype D_0: np.ndarray

    :ivar Omega_climb: 位错攀移激活体积 [m^3]
    :vartype Omega_climb: np.ndarray

    :ivar u_global: 全局坐标系下的1号矢量
    :vartype u_global: np.ndarray

    :ivar v_global: 全局坐标系下的2号矢量
    :vartype v_global: np.ndarray

    :ivar w_global: 全局坐标系下的3号矢量
    :vartype w_global: np.ndarray

    :ivar u_grain: 晶粒坐标系下的1号矢量
    :vartype u_grain: np.ndarray

    :ivar v_grain: 晶粒坐标系下的2号矢量
    :vartype v_grain: np.ndarray

    :ivar w_grain: 晶粒坐标系下的3号矢量
    :vartype w_grain: np.ndarray

    :ivar T: 坐标变换矩阵
    :vartype T: np.ndarray

    :ivar T_voigt: Vogit坐标变换矩阵
    :vartype T_voigt: np.ndarray

    :ivar m_s: 特征滑移系滑移方向
    :vartype m_s: np.ndarray

    :ivar n_s: 特征滑移系滑移面法向
    :vartype n_s: np.ndarray

    :ivar MAX_NITER: 最大迭代次数 [-]
    :vartype MAX_NITER: np.ndarray
    """

    __slots_dict__: dict = {
        'tolerance': ('float', '误差容限'),
        'total_number_of_slips': ('int', '总滑移系数量'),
        'elastic': ('dict', '弹性参数字典'),
        'G': ('float', '剪切模量'),
        'k_b': ('float', '玻尔兹曼常数'),
        'temperature': ('float', '温度'),
        'C': ('np.ndarray', '弹性矩阵'),
        'slip_system_name': ('list[str]', '滑移系统名称'),
        'c_over_a': ('list[float]', '晶体坐标系的c/a'),
        'theta': ('float', '切线系数法参数'),
        'H': ('np.ndarray', '硬化系数矩阵'),
        'tau_sol': ('np.ndarray', '固溶强度'),
        'v_0': ('np.ndarray', '位错滑移速度'),
        'b_s': ('np.ndarray', '位错滑移柏氏矢量长度'),
        'Q_s': ('np.ndarray', '位错滑移激活能'),
        'p_s': ('np.ndarray', '位错滑移阻力拟合参数'),
        'q_s': ('np.ndarray', '位错滑移阻力拟合参数'),
        'd_grain': ('np.ndarray', '平均晶粒尺寸'),
        'i_slip': ('np.ndarray', '平均位错间隔拟合参数'),
        'c_anni': ('np.ndarray', '位错消除拟合参数'),
        'Q_climb': ('np.ndarray', '位错攀移激活能'),
        'D_0': ('np.ndarray', '自扩散系数因子'),
        'Omega_climb': ('np.ndarray', '位错攀移激活体积'),
        'u_global': ('np.ndarray', '全局坐标系下的1号矢量'),
        'v_global': ('np.ndarray', '全局坐标系下的2号矢量'),
        'w_global': ('np.ndarray', '全局坐标系下的3号矢量'),
        'u_grain': ('np.ndarray', '晶粒坐标系下的1号矢量'),
        'v_grain': ('np.ndarray', '晶粒坐标系下的2号矢量'),
        'w_grain': ('np.ndarray', '晶粒坐标系下的3号矢量'),
        'T': ('np.ndarray', '坐标变换矩阵'),
        'T_voigt': ('np.ndarray', 'Vogit坐标变换矩阵'),
        'm_s': ('np.ndarray', '特征滑移系滑移方向'),
        'n_s': ('np.ndarray', '特征滑移系滑移面法向'),
        'MAX_NITER': ('np.ndarray', '最大迭代次数'),
    }

    __slots__ = BaseMaterial.__slots__ + [slot for slot in __slots_dict__.keys()]

    __data_keys__ = ['elastic', 'theta', 'temperature', 'k_b', 'G', 'slip_system_name', 'c_over_a', 'v_0', 'tau_sol', 'b_s', 'Q_s', 'p_s', 'q_s', 'd_grain',
                     'i_slip', 'c_anni', 'Q_climb', 'Omega_climb_coefficient', 'D_0']

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
        self.C: np.ndarray = self.create_elastic_stiffness(self.elastic)
        self.G: float = material.data_dict['G']

        # 物理参数
        self.temperature: float = material.data_dict['temperature']
        self.k_b: float = material.data_dict['k_b']

        # 滑移系参数
        self.total_number_of_slips: int = 0
        self.slip_system_name: list[str] = material.data_dict['slip_system_name']
        self.c_over_a: list[float] = material.data_dict['c_over_a']

        # 多滑移系赋值
        for i, (name, ca) in enumerate(zip(self.slip_system_name, self.c_over_a)):
            slip_system_number, m_s, n_s = generate_mn('slip', name, ca)
            self.total_number_of_slips += slip_system_number
            v_0 = np.ones((slip_system_number,), dtype=DTYPE) * material.data_dict['v_0'][i]
            tau_sol = np.ones((slip_system_number,), dtype=DTYPE) * material.data_dict['tau_sol'][i]
            b_s = np.ones((slip_system_number,), dtype=DTYPE) * material.data_dict['b_s'][i]
            Q_s = np.ones((slip_system_number,), dtype=DTYPE) * material.data_dict['Q_s'][i]
            p_s = np.ones((slip_system_number,), dtype=DTYPE) * material.data_dict['p_s'][i]
            q_s = np.ones((slip_system_number,), dtype=DTYPE) * material.data_dict['q_s'][i]
            d_grain = np.ones((slip_system_number,), dtype=DTYPE) * material.data_dict['d_grain'][i]
            i_slip = np.ones((slip_system_number,), dtype=DTYPE) * material.data_dict['i_slip'][i]
            c_anni = np.ones((slip_system_number,), dtype=DTYPE) * material.data_dict['c_anni'][i]
            Q_climb = np.ones((slip_system_number,), dtype=DTYPE) * material.data_dict['Q_climb'][i]
            D_0 = np.ones((slip_system_number,), dtype=DTYPE) * material.data_dict['D_0'][i]
            Omega_climb = np.ones((slip_system_number,), dtype=DTYPE) * b_s ** 3
            if i == 0:
                self.m_s: np.ndarray = m_s
                self.n_s: np.ndarray = n_s
                self.v_0: np.ndarray = v_0
                self.tau_sol: np.ndarray = tau_sol
                self.b_s: np.ndarray = b_s
                self.Q_s: np.ndarray = Q_s
                self.p_s: np.ndarray = p_s
                self.q_s: np.ndarray = q_s
                self.d_grain: np.ndarray = d_grain
                self.i_slip: np.ndarray = i_slip
                self.c_anni: np.ndarray = c_anni
                self.Q_climb: np.ndarray = Q_climb
                self.D_0: np.ndarray = D_0
                self.Omega_climb: np.ndarray = Omega_climb
            else:
                self.m_s = np.concatenate((self.m_s, m_s))
                self.n_s = np.concatenate((self.n_s, n_s))
                self.v_0 = np.concatenate((self.v_0, v_0))
                self.tau_sol = np.concatenate((self.tau_sol, tau_sol))
                self.b_s = np.concatenate((self.b_s, b_s))
                self.Q_s = np.concatenate((self.Q_s, Q_s))
                self.p_s = np.concatenate((self.p_s, p_s))
                self.q_s = np.concatenate((self.q_s, q_s))
                self.d_grain = np.concatenate((self.d_grain, d_grain))
                self.i_slip = np.concatenate((self.i_slip, i_slip))
                self.c_anni = np.concatenate((self.c_anni, c_anni))
                self.Q_climb = np.concatenate((self.Q_climb, Q_climb))
                self.D_0 = np.concatenate((self.D_0, D_0))
                self.Omega_climb = np.concatenate((self.Omega_climb, Omega_climb))

        self.H: np.ndarray = np.ones(shape=(self.total_number_of_slips, self.total_number_of_slips), dtype=DTYPE)

        # 晶粒取向信息
        self.u_global: np.ndarray = np.array(section.data_dict['u_global'])
        self.v_global: np.ndarray = np.array(section.data_dict['v_global'])
        self.w_global: np.ndarray = np.array(section.data_dict['w_global'])

        self.u_grain: np.ndarray = np.array(section.data_dict['u_grain'])
        self.v_grain: np.ndarray = np.array(section.data_dict['v_grain'])
        self.w_grain: np.ndarray = np.array(section.data_dict['w_grain'])

        self.T: np.ndarray = get_transformation(self.u_grain, self.v_grain, self.w_grain, self.u_global, self.v_global, self.w_global)
        self.T_voigt: np.ndarray = get_voigt_transformation(self.T)

        # 旋转至全局坐标系
        self.m_s = np.dot(self.m_s, self.T)
        self.n_s = np.dot(self.n_s, self.T)
        self.C = np.dot(np.dot(self.T_voigt, self.C), np.transpose(self.T_voigt))

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
            C = np.array([[C11, C12, C12, 0, 0, 0],
                          [C12, C11, C12, 0, 0, 0],
                          [C12, C12, C11, 0, 0, 0],
                          [0, 0, 0, C44, 0, 0],
                          [0, 0, 0, 0, C44, 0],
                          [0, 0, 0, 0, 0, C44]], dtype=DTYPE)
        else:
            raise NotImplementedError(
                error_style(f'the symmetry type \"{symmetry}\" of elastic stiffness is not supported'))
        return C

    def get_tangent(self, variable: dict[str, np.ndarray],
                    state_variable: dict[str, np.ndarray],
                    state_variable_new: dict[str, np.ndarray],
                    element_id: int,
                    iqp: int,
                    ntens: int,
                    ndi: int,
                    nshr: int,
                    timer: Timer) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        r"""
        **获得基于位错演化强化的晶体塑性本构模型**

        本模块中包含3个字典：:py:attr:`variable` ， :py:attr:`state_variable` ， :py:attr:`state_variable_new` 。

        其中，字典 :py:attr:`variable` 存储自由度相关的变量，如应变 :math:`\varepsilon` 和应变增量 :math:`\Delta \varepsilon` 。

        字典 :py:attr:`state_variable` 存储迭代过程中上一个收敛增量步 :math:`t` 时刻的状态变量，如应力 :math:`\sigma` 、分解剪应力 :math:`\tau` 、
        热滑移阻力 :math:`\tau_{pass}` 、剪切应变 :math:`\gamma` 、位错偶极子密度 :math:`\rho_{di}` 、可动位错密度 :math:`\rho_{m}` 、
        特征滑移系滑移方向 :math:`m\_s` 、特征滑移系滑移面法向 :math:`n\_s` 。这些状态变量在当前增量步 :math:`t+\Delta t` 计算收敛之前是不被更新的。

        字典 :py:attr:`state_variable_new` 存储当前增量步 :math:`t+\Delta t` 时刻的某个中间迭代步 :math:`k` 的状态变量。

        ========================================
        基于位错演化强化的晶体塑性本构模型
        ========================================

        目前的晶体塑性建模框架考虑了基于位错滑移的塑性变形机制，其中也涉及温度和应变速率的影响。这些机制包括：

        (a) 位错滑移的热激活流动准则，该规则考虑了温度和应变率对晶体剪切速率的影响；

        (b) 基于位错密度的滑移系统塑性变形强化；

        (c) 基于滑移系统级可动和不可动位错密度的子结构演化；

        (d) 基于物理的滑移系统级背应力演化，该反应力可能有助于循环变形过程中的晶内定向硬化，例如以鲍辛格效应的形式表现出来[1]。

        ----------------------------------------
        1. 引言
        ----------------------------------------

        在晶体塑性本构中通常采用增量形式的本构方程，若全量形式的本构方程采用 :math:`\boldsymbol{\sigma}= \mathbb{C}:{{\boldsymbol{\varepsilon}}}` ，
        其中， :math:`\boldsymbol{\sigma}` 为 Cauchy 应力张量， :math:`\mathbb{C}` 为弹性模量张量， :math:`{{\boldsymbol{\varepsilon}}}` 为应变张量。
        对全量形式的本构方程求时间导数，可得到增量形式的本构方程 :math:`{\boldsymbol{\dot \sigma}}= \mathbb{C}:{{\boldsymbol{D}}}` ，
        其中， :math:`{\boldsymbol{\dot \sigma}}` 为Cauchy应力张量率， :math:`{{\boldsymbol{D}}}` 为变形率张量。
        可以证明，上述增量形式的本构方程中 :math:`{{\boldsymbol{D}}}` 是客观张量，而 :math:`\dot {\boldsymbol{\sigma}}` 不是客观张量。

        要建立正确的材料本构关系必须遵守“客观性公理”这一基本前提。所谓客观性就是物质的力学性质与观察者无关，客观性又称为“标架无差异性”。遵守“客观性公理”的张量称为客观张量，
        也可以定义为在时空变换中保持不变的张量，亦称“时空无差异张量”。用公式可以直接定义满足如下变换率的张量[2]：

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

        选用客观的应力率取代普通的应力率，以保证客观性。本文采用 Cauchy 应力的 Zaremba-Jaumann 率 :math:`\hat{\boldsymbol{\sigma}}` (一般地简称为 Jaumann 率[2,3])。
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
        4. 建立基于位错演化强化的晶体塑性模型
        ----------------------------------------

        在晶体塑性本构模型中，需要通过各滑移系的剪切应变率计算应力率。因此，首先需要确定各滑移系剪切应变的演化方程。
        根据 Orowan 方程[4]，考虑位错演化机制的剪切应变速率 :math:`\dot{\gamma}^{\left( \alpha  \right)}` 由以下因素决定：

        .. math::
            {{\dot \gamma }^{\left( \alpha  \right)}} = \rho _{\text{m}}^{\left( \alpha  \right)}{b_{\text{s}}}{v_0}\exp
            \left\{ { - \frac{{{Q_{\text{s}}}}}{{{k_{\text{b}}}T}}{{\left[ {1 - {{\left\langle {\frac{{\left| {{\tau ^{\left(
            \alpha  \right)}}} \right| - \tau _{{\text{pass}}}^{\left( \alpha  \right)}}}{{\tau _{{\text{sol}}}^{\left( \alpha
            \right)}}}} \right\rangle }^{p_{\text{s}}}}} \right]}^{{q_{\text{s}}}}}} \right\}\operatorname{sign} \left( {{\tau ^{\left( \alpha  \right)}}} \right)

        式中， :math:`\rho^{(\alpha)}_{\rm{m}}` 为滑移系 :math:`\alpha` 的可动位错密度； :math:`\tau^{(\alpha)}` 为滑移系 :math:`\alpha`
        的分解剪应力； :math:`\tau_{\rm{sol}}^{(\alpha)}` 和 :math:`\tau_{\rm{pass}}^{(\alpha)}` 分别为滑移系 :math:`\alpha` 的固溶强度和热滑移阻力。
        另外， :math:`b_{\rm{s}}` 是滑移的Burgers矢量长度， :math:`p_{\text{s}}` 和 :math:`{q_{\text{s}}}` 分别是控制位错滑移阻力曲线的拟合参数
        （取值范围 :math:`0<p \leq 1` ， :math:`1 \leq q \leq 2` ，推荐值 p=0.5, q=1.5）， :math:`v_0` 是参考速度， :math:`Q_{\rm{s}}` 是位错滑移的活化能， :math:`k_{\rm{b}}` 是玻尔兹曼常数。

        其中，热滑移阻力 :math:`\tau_{\rm{pass}}^{(\alpha)}` 与位错密度行为相关：

        .. math::
            \tau _{{\text{pass}}}^{\left( \alpha  \right)} = G{b_{\text{s}}}\sqrt {\sum\limits_{\beta  = 1}^N {{h_{\alpha
            \beta }}} \left( {\rho _{\text{m}}^{\left( \beta  \right)} + \rho _{{\text{di}}}^{\left( \beta  \right)}} \right)}

        式子中， :math:`G` 是剪切模量， :math:`N_{\rm{s}}` 是总滑移系统数， :math:`h_{\alpha \beta}` 是滑移系统 :math:`\alpha` 和 :math:`\beta` 相互作用的系数矩阵；
        滑移系统 :math:`\beta` 上的总位错密度 :math:`\rho^{(\beta)}` 由移动位错密度 :math:`\rho^{(\beta)}_{\rm{m}}` 和位错偶极子密度 :math:`\rho^{(\beta)a}_{\rm{di}}` 组成。

        将可动位错密度 :math:`\rho^{(\alpha)}_{\rm{m}}` 和位错偶极子密度 :math:`\rho^{(\alpha)}_{\rm{di}}` 相加定义为 :math:`\rho^{(\alpha)}` ：

        .. math::
            {\rho ^{\left( \alpha  \right)}}{\text{ = }}\rho _{\text{m}}^{\left( \alpha  \right)} + \rho _{{\text{di}}}^{\left( \alpha  \right)}

        则热滑移阻力 :math:`\tau_{\rm{pass}}^{(\alpha)}` 的表达式可以改写为：

        .. math::
            \tau _{{\text{pass}}}^{\left( \alpha  \right)} = G{b_{\text{s}}}\sqrt {\sum\limits_{\beta  = 1}^N {{h_{\alpha
            \beta }}} \left( {\rho _{\text{m}}^{\left( \beta  \right)} + \rho _{{\text{di}}}^{\left( \beta  \right)}} \right)} =
            G{b_{\text{s}}}\sqrt {\sum\limits_{\beta  = 1}^N {{h_{\alpha \beta }}} {\rho ^{\left( \beta  \right)}}}

        进一步，将热滑移阻力 :math:`\tau_{\rm{pass}}^{(\alpha)}` 对时间求导，得到其演化方程：

        .. math::
            \dot \tau _{{\text{pass}}}^{\left( \alpha  \right)} = G{b_{\text{s}}}\frac{1}{{2\sqrt {\sum\limits_{\beta
            = 1}^N {{h_{\alpha \beta }}} {\rho ^{\left( \beta  \right)}}} }}\sum\limits_{\beta  = 1}^N {{h_{\alpha \beta }}}
            {{\dot \rho }^{\left( \beta  \right)}} = \frac{{{{\left( {G{b_{\text{s}}}} \right)}^2}}}{{2\tau _{{\text{pass}}}^{
            \left( \alpha  \right)}}}\sum\limits_{\beta  = 1}^N {{h_{\alpha \beta }}} {{\dot \rho }^{\left( \beta  \right)}}

        根据 Blum 和 Eisenlohr[5] 以及 Ma 和 Roters[6] 的研究，位错密度的演变是位错生长、湮灭和偶极子形成的综合结果。对于可动位错来说，
        它们的湮灭是由两个Burgers矢量方向相反的位错相互靠近或形成偶极子造成的。它们的演化率如下：

        位错增殖：

        .. math::
            \dot{\rho}_{\rm{multip}}^{\left( \alpha  \right)} = \frac{\left|\dot{\gamma}^{\left( \alpha  \right)}\right|}{b_{\rm{s}} \lambda^{\left( \alpha  \right)}}

        位错偶极子行成：

        .. math::
            \dot{\rho}_{\rm{diform}}^{\left( \alpha  \right)} = \frac{2 \rho_{\rm{m}}^{\left( \alpha  \right)} \left|\dot{\gamma}^{\left(
            \alpha  \right)}\right|}{b_{\rm{s}} }\left(d_{\rm{di}}^{\left( \alpha  \right)}-d_{\rm{min}}^{\left( \alpha  \right)} \right)

        位错湮灭：

        .. math::
            \dot{\rho}_{\rm{anni}}^{\left( \alpha  \right)} = \frac{2 \rho_{\rm{m}}^{\left( \alpha  \right)} \left|\dot{\gamma}^{\left( \alpha  \right)}\right|}
            {b_{\rm{s}} }d_{\rm{min}}^{\left( \alpha  \right)}

        对于可动位错：

        .. math::
            \dot{\rho}_{\rm{m}}^{\left( \alpha  \right)} = \frac{\left|\dot{\gamma}^{\left( \alpha  \right)} \right|}{b_{\rm{s}} \lambda^{\left( \alpha  \right)}}-\frac{2
            \rho^{\left( \alpha  \right)}_{\rm{m}} \left|\dot{\gamma}^{\left( \alpha  \right)}\right|}{b_{\rm{s}}}d^{\left( \alpha  \right)}_{\rm{di}}

        而对于位错偶极子来说，位错密度降低是由异号位错湮灭或攀移引起的：

        .. math::
            \dot{\rho}^{\left( \alpha  \right)}_{\rm{di}}=\frac{2\rho^{\left( \alpha  \right)}_{\rm{m}} \left|\dot{\gamma}^{\left( \alpha  \right)}\right|}{b_{\rm{s}}}
            \left(d_{\rm{di}}^{\left( \alpha  \right)}-d_{\rm{min}}^{\left( \alpha  \right)} \right) -\frac{2\rho^{\left( \alpha  \right)}_{\rm{di}} \left|\dot{\gamma}^{\left( \alpha  \right)}
            \right|}{b_{\rm{s}}} d^{\left( \alpha  \right)}_{\rm{min}} - \frac{4 \rho^{\left( \alpha  \right)}_{\rm{di}} \nu_{\rm{clim}}^{\left( \alpha  \right)}}{d^{\left( \alpha  \right)}_{\rm{di}}-d^{\left( \alpha  \right)}_{\rm{min}}}

        其中， :math:`{{\lambda ^{\left( \alpha  \right)}}}` 是位错滑移的平均自由程，它受到位错与晶界和其他位错相互作用的影响； :math:`d_{{\text{di}}}^{\left( \alpha  \right)}`
        是两个位错可形成稳定偶极子的滑移面最大间距； :math:`d_{{\text{min}}}^{\left( \alpha  \right)}` 是两个异号位错湮灭的最小间距； :math:`\nu _{{\text{clim}}}^{\left( \alpha  \right)}` 是位错攀移速度。
        这些变量的计算公式如下：

        .. math::
            \frac{1}{{{\lambda ^{\left( \alpha  \right)}}}} = \frac{1}{d} + \frac{1}{{{i_{{\text{slip}}}}}}\sqrt
            {\sum\limits_{\beta  = 1}^N {{h_{\alpha \beta }}} {\rho ^{\left( \beta  \right)}}}  = \frac{1}{d} +
            \frac{1}{{{i_{{\text{slip}}}}}}\frac{{\tau _{{\text{pass}}}^{\left( \alpha  \right)}}}{{G{b_{\text{s}}}}}

        .. math::
            d_{{\text{di}}}^{\left( \alpha  \right)} = \frac{{3G{b_{\text{s}}}}}{{16\pi \left| {{\tau ^{\left( \alpha  \right)}}} \right|}}

        .. math::
            d_{{\text{min}}}^{\left( \alpha  \right)} = {c_{{\text{anni}}}}{b_{\text{s}}}

        .. math::
            \nu _{{\text{clim}}}^{\left( \alpha  \right)} = \frac{{3G{D_0}{\Omega _{{\text{clim}}}}}}{{2\pi {k_{\text{b}}}T
            (d_{{\text{di}}}^{\left( \alpha  \right)} + d_{{\text{min}}}^{\left( \alpha  \right)})}}\exp ( -
            \frac{{{Q_{{\text{clim}}}}}}{{{k_{\text{b}}}T}})

        其中， :math:`d` 为平均晶粒尺寸； :math:`{{D_0}}` 为材料的自扩散系数（受空位机制控制）； :math:`{{\Omega _{{\text{clim}}}}}` 和 :math:`Q_{\rm{clim}}`
        分别为位错攀移激活体积和激活能； :math:`i_{\rm{slip}}` 为平均位错滑移间距系数； :math:`c_{\rm{anni}}` 为位错湮灭的拟合参数。

        最后，结合公式 :math:`\dot \rho _{\text{m}}^{\left( \alpha  \right)}` ,  :math:`\frac{1}{{{\lambda ^{\left( \alpha  \right)}}}}`
        和公式 :math:`d_{{\text{di}}}^{\left( \alpha  \right)}` ，得到可动位错密度演化为：

        .. math::
            \dot \rho _{\text{m}}^{\left( \alpha  \right)} = \frac{{\left| {{{\dot \gamma }^{\left( \alpha  \right)}}}
            \right|}}{{{b_{\text{s}}}{\lambda ^{\left( \alpha  \right)}}}} - \frac{{2\rho _{\text{m}}^{\left( \alpha
            \right)}\left| {{{\dot \gamma }^{\left( \alpha  \right)}}} \right|}}{{{b_{\text{s}}}}}d_{{\text{di}}}^{\left( \alpha  \right)}

        结合公式 :math:`\dot \rho _{\text{m}}^{\left( \alpha  \right)}` ， :math:`\frac{1}{{{\lambda ^{\left( \alpha  \right)}}}}` 和
        公式 :math:`d_{{\text{di}}}^{\left( \alpha  \right)}` 、 :math:`d_{{\text{min}}}^{\left( \alpha  \right)}`
        和  :math:`\nu _{{\text{clim}}}^{\left( \alpha  \right)}` 位错偶极子密度演化为：

        .. math::
            \dot \rho _{{\text{di}}}^{\left( \alpha  \right)} = \frac{{2\rho _{\text{m}}^{\left( \alpha  \right)}\left|
            {{{\dot \gamma }^{\left( \alpha  \right)}}} \right|}}{{{b_{\text{s}}}}}\left( {d_{{\text{di}}}^{\left( \alpha
            \right)} - d_{{\text{min}}}^{\left( \alpha  \right)}} \right) - \frac{{2\rho _{{\text{di}}}^{\left( \alpha
            \right)}\left| {{{\dot \gamma }^{\left( \alpha  \right)}}} \right|}}{{{b_{\text{s}}}}}d_{{\text{min}}}^{\left(
            \alpha  \right)} - \frac{{4\rho _{{\text{di}}}^{\left( \alpha  \right)}\nu _{{\text{clim}}}^{\left( \alpha
            \right)}}}{{d_{{\text{di}}}^{\left( \alpha  \right)} - d_{{\text{min}}}^{\left( \alpha  \right)}}}

        接下来，求解 :math:`\tau^{(\alpha)}` 的演化方程 :math:`\dot{\tau}^{(\alpha)}` 。

        滑移系上的分解剪应力 :math:`{\tau ^{\left( \alpha  \right)}}` 定义为：

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

        通过上述推导，方程 :math:`{{\dot \gamma }^{\left( \alpha  \right)}}` 建立了能够描述晶体循环变形中位错演化强化的剪切应变硬化方程。
        利用计算得到的各滑移系中的剪切应变增量和晶体塑性理论中的本构关系，即可得到宏观应力增量，下面将详细介绍位错演化模型的数值离散过程。

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
            \right)}}\left( t \right) + {\left. {\frac{{\partial {{\dot \gamma }^{\left( \alpha  \right)}}}}{{\partial
            \rho _{\text{m}}^{\left( \alpha  \right)}}}} \right|_t}\Delta \rho _{\text{m}}^{\left( \alpha  \right)} +
            {\left. {\frac{{\partial {{\dot \gamma }^{\left( \alpha  \right)}}}}{{\partial {\tau ^{\left( \alpha
            \right)}}}}} \right|_t}\Delta {\tau ^{\left( \alpha  \right)}} + {\left. {\frac{{\partial {{\dot \gamma }^{\left(
            \alpha  \right)}}}}{{\partial \tau _{{\text{pass}}}^{\left( \alpha  \right)}}}} \right|_t}\Delta
            \tau_{{\text{pass}}}^{\left( \alpha  \right)}

        可得剪切应变的离散格式 :math:`\Delta \gamma^{(\alpha)}` ：

        .. math::
            \Delta {\gamma ^{\left( \alpha  \right)}} = \Delta t\left[ {{{\dot \gamma }^{\left( \alpha  \right)}}\left( t
            \right) + {{\left. {\theta \frac{{\partial {{\dot \gamma }^{\left( \alpha  \right)}}}}{{\partial \rho _{\text{m}}^{\left(
            \alpha  \right)}}}} \right|}_t}\Delta \rho _{\text{m}}^{\left( \alpha  \right)} + \theta {{\left. {\frac{{\partial {{\dot
            \gamma }^{\left( \alpha  \right)}}}}{{\partial {\tau ^{\left( \alpha  \right)}}}}} \right|}_t}\Delta {\tau ^{\left( \alpha
            \right)}} + {{\left. {\theta \frac{{\partial {{\dot \gamma }^{\left( \alpha  \right)}}}}{{\partial \tau _{{\text{pass}}}^{\left(
            \alpha  \right)}}}} \right|}_t}\Delta \tau _{{\text{pass}}}^{\left( \alpha  \right)}} \right]

        对 :math:`\dot \tau _{{\text{pass}}}^{\left( \alpha  \right)}` 进行积分，得到热滑移阻力的离散格式：

        .. math::
            \Delta \tau _{{\text{pass}}}^{\left( \alpha  \right)} = \frac{{{{\left( {G{b_{\text{s}}}} \right)}^2}}}{{2
            \tau _{{\text{pass}}}^{\left( \alpha  \right)}}}\sum\limits_{\beta  = 1}^N {{h_{\alpha \beta }}} \Delta {\rho ^{\left( \beta  \right)}}

        对 :math:`\dot \rho _{\text{m}}^{\left( \alpha  \right)}` 进行积分，得到可动位错密度离散格式：

        .. math::
            \Delta \rho _{\text{m}}^{\left( \alpha  \right)} = \left[ {\frac{1}{{{b_{\text{s}}}{\lambda ^{\left( \alpha  \right)}}}}
            - \frac{{2\rho _{\text{m}}^{\left( \alpha  \right)}d_{{\text{di}}}^{\left( \alpha  \right)}}}{{{b_{\text{s}}}}}}
            \right]\left| {\Delta {\gamma ^{\left( \alpha  \right)}}} \right|

        对 :math:`\dot \rho _{{\text{di}}}^{\left( \alpha  \right)}` 进行积分，得到位错偶极子密度离散格式：

        .. math::
            \Delta \rho _{di}^{\left( \alpha  \right)} = 2\left[ {\rho _{\text{m}}^{\left( \alpha  \right)}d_{{\text{di}}}^{\left(
            \alpha  \right)} - \left( {\rho _{\text{m}}^{\left( \alpha  \right)} + \rho _{{\text{di}}}^{\left( \alpha  \right)}}
            \right)d_{{\text{min}}}^{\left( \alpha  \right)}} \right]\frac{{\left| {\Delta {\gamma ^{\left( \alpha  \right)}}}
            \right|}}{{{b_{\text{s}}}}} - \frac{{4\rho _{{\text{di}}}^{\left( \alpha  \right)}\nu _{{\text{clim}}}^{\left( \alpha
            \right)}}}{{d_{{\text{di}}}^{\left( \alpha  \right)} - d_{{\text{min}}}^{\left( \alpha  \right)}}}\Delta t

        对上式进行化简：

        .. math::
            \Delta \rho _{di}^{\left( \alpha  \right)} = 2\left[ {\rho _{\text{m}}^{\left( \alpha  \right)}d_{{\text{di}}}^{\left( \alpha  \right)} - {\rho ^{\left(
            \alpha  \right)}}d_{{\text{min}}}^{\left( \alpha  \right)}} \right]\frac{{\left| {\Delta {\gamma ^{\left(
            \alpha  \right)}}} \right|}}{{{b_{\text{s}}}}} - \frac{{4\rho _{{\text{di}}}^{\left( \alpha  \right)}
            \nu _{{\text{clim}}}^{\left( \alpha  \right)}}}{{d_{{\text{di}}}^{\left( \alpha  \right)} - d_{{\text{min}}}^{\left(
            \alpha  \right)}}}\Delta t

        将位错偶极子密度与可动位错密度的离散格式相加，得到位错密度离散格式：

        .. math::
            \Delta {\rho ^{\left( \alpha  \right)}} = \Delta \rho _{{\text{di}}}^{\left( \alpha  \right)} + \Delta
            \rho _{\text{m}}^{\left( \alpha  \right)} = \left[ {\frac{1}{{{b_{\text{s}}}{\lambda ^{\left( \alpha
            \right)}}}} - 2d_{{\text{min}}}^{\left( \alpha  \right)}\frac{{{\rho ^{\left( \alpha  \right)}}}}
            {{{b_{\text{s}}}}}} \right]\left| {\Delta {\gamma ^{\left( \alpha  \right)}}} \right| - \frac{{4\rho _{{
            \text{di}}}^{\left( \alpha  \right)}\nu _{{\text{clim}}}^{\left( \alpha  \right)}}}{{d_{{\text{di}}}^{\left(
            \alpha  \right)} - d_{{\text{min}}}^{\left( \alpha  \right)}}}\Delta t

        接下来，对 :math:`{\dot \tau }` 进行积分，得到分解剪应力离散格式：

        .. math::
            \Delta \tau^{(\alpha)} = {{\dot \tau }^{\left( \alpha  \right)}} \Delta t=\boldsymbol{S}^{(\alpha)}: \Delta
            \boldsymbol{\varepsilon}-\boldsymbol {S}^{(\alpha)}: \sum_{\beta=1}^{N} \boldsymbol {P}^{(\beta)} \Delta \gamma^{(\beta)}
            =  \boldsymbol{S}^{(\alpha)}: \left(\Delta \boldsymbol{\varepsilon}- \sum_{\beta=1}^{N} \boldsymbol {P}^{(\beta)} \Delta
            \gamma^{(\beta)} \right)

        令：

        .. math::
            X^{\left( \alpha  \right)} = \frac{{\left| {{\tau ^{\left( \alpha  \right)}}} \right| - \tau _{{\text{pass}}}^{\left( \alpha
            \right)}}}{{\tau _{{\text{sol}}}^{\left( \alpha  \right)}}}, {A_s} = \frac{{{Q_{\text{s}}}}}{{{k_{\text{b}}}T}}

        得到位错演化强化模型：

        .. math::
           {{\dot \gamma }^{\left( \alpha  \right)}} = \rho _{\text{m}}^{\left( \alpha  \right)}{b_{\text{s}}}{v_0}\exp
           \left[ { - {A_s}{{\left( {1 - {{\left\langle X^{\left( \alpha  \right)} \right\rangle }^{p_{\text{s}}}}} \right)}^{q_{\text{s}}}}} \right]\operatorname{sign}
           \left( {{\tau ^{\left( \alpha  \right)}}} \right)

        接下来采用链式法则，对每一项分别求导：

        .. math::
            \frac{{\partial {{\dot \gamma }^{\left( \alpha  \right)}}}}{{\partial \left\langle X^{\left( \alpha  \right)} \right\rangle }} =
            {A_s}\rho _m^{\left( \alpha  \right)}{b_s}pq{v_0}{\left\langle X^{\left( \alpha  \right)} \right\rangle ^{p_{\text{s}} - 1}}{\left( {1 -
            {{\left\langle X^{\left( \alpha  \right)} \right\rangle }^{p_{\text{s}}}}} \right)^{{q_{\text{s}}} - 1}}{e^{ - {A_s}{{\left( {1 - {{\left\langle X^{\left( \alpha  \right)}
            \right\rangle }^{p_{\text{s}}}}} \right)}^{q_{\text{s}}}}}}\operatorname{sign} \left( {{\tau ^{\left( \alpha  \right)}}} \right)

        .. math::
            \frac{{\partial \left\langle X^{\left( \alpha  \right)} \right\rangle }}{{\partial {\tau ^{\left( \alpha  \right)}}}} =
            \frac{1}{{\tau _{{\text{sol}}}^{\left( \alpha  \right)}}}\operatorname{sign} \left( {{\tau ^{\left(
            \alpha  \right)}}} \right)H\left( X^{\left( \alpha  \right)} \right)

        .. math::
            \frac{{\partial \left\langle X^{\left( \alpha  \right)} \right\rangle }}{{\partial \tau _{{\text{pass}}}^{\left( \alpha  \right)}}} =
            - \frac{1}{{\tau _{{\text{sol}}}^{\left( \alpha  \right)}}}H\left( X^{\left( \alpha  \right)} \right)

        .. math::
            \frac{{\partial {{\dot \gamma }^{\left( \alpha  \right)}}}}{{\partial \rho _{\text{m}}^{\left( \alpha \right)}}}
            = {b_{\text{s}}}{v_0}\exp \left\{ { - {A_s}{{\left[ {1 - {{\left\langle X^{\left( \alpha  \right)} \right\rangle }^{p_{\text{s}}}}} \right]}^{q_{\text{s}}}}}
            \right\}\operatorname{sign} \left( {{\tau ^{\left( \alpha  \right)}}} \right)

        所以有：

        .. math::
            \Delta {\gamma ^{\left( \alpha  \right)}} = \Delta t{{\dot \gamma }^{\left( \alpha  \right)}}\left( t \right)
            + \Delta t\theta \left( {\frac{{\partial {{\dot \gamma }^{\left( \alpha  \right)}}}}{{\partial \left\langle X^{\left( \alpha  \right)}
            \right\rangle }}\frac{{\partial \left\langle X^{\left( \alpha  \right)} \right\rangle }}{{\partial \Delta \rho _{\text{m}}^{\left(
            \alpha  \right)}}}\Delta \rho _{\text{m}}^{\left( \alpha  \right)} + \frac{{\partial {{\dot \gamma }^{\left(
            \alpha  \right)}}}}{{\partial \left\langle X^{\left( \alpha  \right)} \right\rangle }}\frac{{\partial \left\langle X^{\left( \alpha  \right)} \right\rangle }}
            {{\partial {\tau ^{\left( \alpha  \right)}}}}\Delta {\tau ^{\left( \alpha  \right)}} + \frac{{\partial {{\dot
            \gamma }^{\left( \alpha  \right)}}}}{{\partial \left\langle X^{\left( \alpha  \right)} \right\rangle }}\frac{{\partial \left\langle X^{\left( \alpha  \right)}
            \right\rangle }}{{\partial \Delta \tau _{{\text{pass}}}^{\left( \alpha  \right)}}}\Delta
            \tau_{{\text{pass}}}^{\left( \alpha  \right)}} \right)

        将 :math:`\Delta \rho _{\text{m}}^{\left( \alpha  \right)}` ， :math:`\Delta \tau _{{\text{pass}}}^{\left( \alpha  \right)}` ， :math:`\Delta \tau^{(\alpha)}`
        代入 :math:`\Delta \gamma^{(\alpha)}` ，再将其写成矩阵形式，可得：

        .. math::
            \left[ {\begin{array}{*{20}{l}}
            {{\delta _{\alpha \beta }}} \\
            { - term1 \cdot term2 \cdot term5 \cdot {b_{\text{s}}}{v_0}\exp \left\{ { - {A_s}{{\left[ {1 - {{\left\langle X^{\left( \alpha  \right)} \right\rangle }^{p_{\text{s}}}}} \right]}^{q_{\text{s}}}}} \right\}{\delta _{\alpha \beta }}} \\
            { + term1 \cdot term2 \cdot term3 \cdot term4 \cdot \operatorname{sign} \left( {{\tau ^{\left( \alpha  \right)}}} \right)} \\
            { + term1 \cdot term2 \cdot term3 \cdot term8 \cdot {h_{\alpha \beta }}\left[ {{\delta _{\alpha \beta }} \cdot term6 \cdot {\text{sign}}\left( {\Delta {\tau ^{\left( \beta  \right)}}} \right)} \right]}
            \end{array}} \right]\Delta {\gamma ^{\left( \alpha  \right)}}
            = \Delta t{{\dot \gamma }^{\left( \alpha  \right)}}\left( t \right) + term1 \cdot term2 \cdot term3 \cdot \operatorname{sign}
            \left( {{\tau ^{\left( \alpha  \right)}}} \right) \cdot {{\boldsymbol{S}}^{\left( \alpha  \right)}}:\Delta {\boldsymbol{\varepsilon }} +
            term1 \cdot term2 \cdot term3 \cdot term8 \cdot \sum\limits_{\beta  = 1}^N {{h_{\alpha \beta }} \cdot term7}

        整理得到引入位错演化强化的晶体塑性模型迭代格式为由 :math:`N` 个末知数 :math:`\Delta \gamma^{(\alpha)}` 和 :math:`N` 个非线性方程组成的方程组。
        其中 :math:`term1-8` 是编程过程中为了简化计算引入的一些临时变量：

        .. math::
            term1 := \Delta t \theta

        .. math::
            term2 := {A_s}\rho _m^{\left( \alpha  \right)}{b_s}pq{v_0}{\left\langle X^{\left( \alpha  \right)} \right\rangle ^{{p_{\text{s}}} - 1}}{\left(
            {1 - {{\left\langle X^{\left( \alpha  \right)} \right\rangle }^{p_{\text{s}}}}} \right)^{{q_{\text{s}}} - 1}}{e^{ - {A_s}{{\left( {1 - {{\left\langle X^{\left( \alpha  \right)}
            \right\rangle }^{p_{\text{s}}}}} \right)}^{q_{\text{s}}}}}}\operatorname{sign} \left( {{\tau ^{\left( \alpha  \right)}}} \right)

        .. math::
            term3 := \frac{{H\left( X^{\left( \alpha  \right)} \right)}}{{\tau _{{\text{sol}}}^{\left( \alpha  \right)}}}

        .. math::
            term4 := {{\boldsymbol{S}}^{\left( \alpha  \right)}}:{{\boldsymbol{P}}^{\left( \beta  \right)}}

        .. math::
            term5 := \frac{1}{{{b_{\text{s}}}{\lambda ^{\left( \alpha  \right)}}}} - 2d_{{\text{di}}}^{\left( \alpha
            \right)}\frac{{\rho _{\text{m}}^{\left( \alpha  \right)}}}{{{b_{\text{s}}}}}

        .. math::
            term6 := \frac{1}{{{b_{\text{s}}}{\lambda ^{\left( \alpha  \right)}}}} - 2d_{\min }^{\left( \alpha
            \right)}\frac{{{\rho ^{\left( \alpha  \right)}}}}{{{b_{\text{s}}}}}

        .. math::
            term7 := \frac{{4\rho _{{\text{di}}}^{\left( \beta  \right)}\nu _{{\text{clim}}}^{\left( \beta
            \right)}}}{{d_{{\text{di}}}^{\left( \beta  \right)} - d_{{\text{min}}}^{\left( \beta  \right)}}}\Delta t

        .. math::
            term8 := \frac{{{{\left( {G{b_{\text{s}}}} \right)}^2}}}{{2\tau _{{\text{pass}}}^{\left( \alpha  \right)}}}

        上标 :math:`\alpha` 表示第 :math:`\alpha` 个滑移系( :math:`\alpha=1 \sim N，N` 为所有可能开动滑移系的数目)，
        等式右边的 :math:`\Delta {\boldsymbol{\varepsilon }}` 为已知项。求解该非线性方程组可以得到所有滑移系的初始剪切应变增量 :math:`\Delta \gamma^{(\alpha)}`，
        进而计算应力增量 :math:`\Delta \boldsymbol{\sigma}` 和其他状态变量的初始增量。

        将上面的方程组简写为以下形式：

        .. math::
            \boldsymbol {A} \Delta {\gamma ^{\left( \alpha  \right)}} = \Delta t{{\dot \gamma }^{\left( \alpha  \right)}}
            \left( t \right) + term1\cdot term2\cdot term3\cdot \operatorname{sign} \left( {{\tau ^{\left( \alpha  \right)}}} \right)
            {{\boldsymbol{S}}^{\left( \alpha  \right)}}:\Delta {\boldsymbol{\varepsilon }} + term1\cdot term2\cdot term3\cdot term8\cdot \sum\limits_{\beta
            = 1}^N {{h_{\alpha \beta }}\cdot term7}

        方程两边对 :math:`\Delta {\boldsymbol{\varepsilon }}` 求偏导得到：

        .. math::
            \boldsymbol {A} \frac{{\partial \Delta {\gamma ^{\left( \alpha  \right)}}}}{{\partial \Delta {\boldsymbol{\varepsilon }}}} =
            term1 \cdot term2 \cdot term3 \cdot \operatorname{sign} \left( {{\tau ^{\left( \alpha  \right)}}} \right) \cdot
            {{\boldsymbol{S}}^{\left( \alpha  \right)}}

        所以有：

        .. math::
            ddgdde = \frac{{\partial \Delta {\gamma ^{\left( \alpha  \right)}}}}{{\partial \Delta {\boldsymbol{\varepsilon }}}} =
            {{\boldsymbol {A}}^{ - 1}} \cdot term1 \cdot term2\cdot term3\cdot \operatorname{sign} \left( {{\tau ^{\left( \alpha  \right)}}} \right) \cdot
            {{\boldsymbol{S}}^{\left( \alpha  \right)}}

        在第3节，我们已经得到 Jaumann率 表达式为：

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
            {\left\{ {{X^{\left( \alpha  \right)}}} \right\}^{\left( {k + 1} \right)}} = \frac{{\left| {\tau _t^{\left( \alpha  \right)} +
            {{\left\{ {\Delta {\tau ^{\left( \alpha  \right)}}} \right\}}^{\left( {k + 1} \right)}}} \right| - \left(
            {\tau _{{\text{pass}},t}^{\left( \alpha  \right)} + {{\left\{ {\Delta \tau _{{\text{pass}}}^{\left( \alpha
            \right)}} \right\}}^{\left( {k + 1} \right)}}} \right)}}{{\tau _{{\text{sol}}}^{\left( \alpha  \right)}}}

        进而可得到即将用于牛顿拉夫森迭代的剪切应变速率表达式：

        .. math::
            {\left\{ {\dot \gamma _{t + \Delta t}^{\left( \alpha  \right)}} \right\}^{\left( {k + 1} \right)}} = \left( {\rho _{{\text{m}},t}^{\left( \alpha  \right)} +
            {{\left\{ {\Delta \rho _{\text{m}}^{\left( \alpha  \right)}} \right\}}^{\left( {k + 1} \right)}}} \right){b_{\text{s}}}{v_0}
            \exp \left\{ { - {A_{\text{s}}}{{\left[ {1 - {{\left\langle {{{\left\{ {\Delta {\tau ^{\left( \alpha  \right)}}} \right\}}^{\left( {k + 1} \right)}}}
            \right\rangle }^{{p_{\text{s}}}}}} \right]}^{{q_{\text{s}}}}}} \right\}\operatorname{sign} \left( {\tau _t^{\left( \alpha  \right)} +
            {{\left\{ {\Delta {\tau ^{\left( \alpha  \right)}}} \right\}}^{\left( {k + 1} \right)}}} \right)

        参考 5.1 节的推导，最后我们得到迭代求解该非线性方程组的所有滑移系的剪切应变增量的方程组：

        .. math::
            {\boldsymbol{A}}^{\left( k + 1 \right)} \left( {{{\left\{ {\Delta {\gamma ^{\left( \alpha  \right)}}} \right\}}^{\left( k \right)}}} \right)
            \cdot \Delta {\left\{ {\Delta {\gamma ^{\left( \alpha  \right)}}} \right\}^{\left( {k + 1} \right)}} = \left\{ {rhs} \right\}^{\left( {k} \right)}

        其中，刚度矩阵 :math:`{\boldsymbol{A}}^{\left( k + 1 \right)} \left( {{{\left\{ {\Delta {\gamma ^{\left( \alpha  \right)}}} \right\}}^{\left( k \right)}}} \right)` 为：

        .. math::
           {\boldsymbol{A}}^{\left( k + 1 \right)}  =
            F'\left( {{{\left\{ {\Delta {\gamma ^{\left( \alpha  \right)}}} \right\}}^{\left( k \right)}}} \right) = \left[ {\begin{array}{*{20}{l}}
              {{\delta _{\alpha \beta }}} \\
              { - term1 \cdot term2 \cdot term5 \cdot {b_{\text{s}}}{v_0}\exp \left\{ { - {A_s}{{\left[ {1 - {{\left\langle {{{\left\{ {{X^{\left( \alpha  \right)}}} \right\}}^{\left( {k + 1} \right)}}} \right\rangle }^{{p_{\text{s}}}}}} \right]}^{{q_{\text{s}}}}}} \right\}{\delta _{\alpha \beta }}} \\
              { + term1 \cdot term2 \cdot term3 \cdot term4 \cdot \operatorname{sign} \left( {{{\left\{ {{\tau ^{\left( \alpha  \right)}}} \right\}}^{\left( {k + 1} \right)}}} \right)} \\
              { + term1 \cdot term2 \cdot term3 \cdot term8 \cdot {h_{\alpha \beta }}\left[ {term6 \cdot {\text{sign}}\left( {{{\left\{ {{\tau ^{\left( \alpha  \right)}}} \right\}}^{\left( {k + 1} \right)}}} \right) \cdot {\delta _{\alpha \beta }}} \right]}
            \end{array}} \right]

        可以看出，用于迭代的刚度矩阵 :math:`{\boldsymbol{A}}^{\left( k + 1 \right)}` 与求解初值的刚度矩阵 :math:`{\boldsymbol{A}}` 形式一致。

        其中，临时变量 :math:`term1-8` 为：

        .. math::
            term1 := \Delta t \theta

        .. math::
            term2: = {A_s}{\left\{ {\Delta \rho _{\text{m}}^{\left( \alpha  \right)}} \right\}^{\left( {k + 1} \right)}}{b_s}pq{v_0}{\left\langle {{{\left\{ {{X^{\left( \alpha  \right)}}} \right\}}^{\left( {k + 1} \right)}}} \right\rangle ^{{p_{\text{s}}} - 1}}{\left( {1 - {{\left\langle {{{\left\{ {{X^{\left( \alpha  \right)}}} \right\}}^{\left( {k + 1} \right)}}} \right\rangle }^{{p_{\text{s}}}}}} \right)^{{q_{\text{s}}} - 1}}{e^{ - {A_s}{{\left( {1 - {{\left\langle {{{\left\{ {{X^{\left( \alpha  \right)}}} \right\}}^{\left( {k + 1} \right)}}} \right\rangle }^{{p_{\text{s}}}}}} \right)}^{{q_{\text{s}}}}}}}{\text{sign}}\left( {{{\left\{ {{\tau ^{\left( \alpha  \right)}}} \right\}}^{\left( {k + 1} \right)}}} \right)

        .. math::
            term3 := \frac{{H\left( {{{\left\{ {{X^{\left( \alpha  \right)}}} \right\}}^{\left( {k + 1} \right)}}} \right)}}{{\tau _{{\text{sol}}}^{\left( \alpha  \right)}}}

        .. math::
            term4 := {{\boldsymbol{S}}^{\left( \alpha  \right)}}:{{\boldsymbol{P}}^{\left( \beta  \right)}}

        .. math::
            term5: = \frac{1}{{{b_{\text{s}}}{{\left\{ {{\lambda ^{\left( \alpha  \right)}}} \right\}}^{\left( {k + 1} \right)}}}} -
            2d_{{\text{di}}}^{\left( \alpha  \right)}\frac{{{{\left\{ {\rho _{\text{m}}^{\left( \alpha  \right)}} \right\}}^{\left( {k + 1} \right)}}}}{{{b_{\text{s}}}}}

        .. math::
            term6: = \frac{1}{{{b_{\text{s}}}{{\left\{ {{\lambda ^{\left( \alpha  \right)}}} \right\}}^{\left( {k + 1} \right)}}}} - 2{d_{{{\min }^{\left( \alpha
            \right)}}}}\frac{{{{\left\{ {{\rho ^{\left( \alpha  \right)}}} \right\}}^{\left( {k + 1} \right)}}}}{{{b_{\text{s}}}}}

        .. math::
            term7: = \frac{{4{{\left\{ {\rho _{{\text{di}}}^{\left( \beta  \right)}} \right\}}^{\left( {k + 1} \right)}}\nu _{{\text{clim}}}^{\left( \beta
            \right)}}}{{d_{{\text{di}}}^{\left( \beta  \right)} - d_{{\text{min}}}^{\left( \beta  \right)}}}\Delta t

        .. math::
            term8: = \frac{{{{\left( {G{b_{\text{s}}}} \right)}^2}}}{{2{{\left\{ {\tau _{{\text{pass}}}^{\left( \alpha  \right)}} \right\}}^{\left( {k + 1} \right)}}}}

        方程组右边项 :math:`rhs` 为：

        .. math::
            \left\{ {rhs} \right\}^{\left( {k} \right)} =  - F\left( {{{\left\{ {\Delta {\gamma ^{\left( \alpha  \right)}}} \right\}}^{\left( k \right)}}} \right)
            = \Delta t\left( {1 - \theta } \right){{\dot \gamma }^{\left( \alpha  \right)}}\left( t \right) +
            \Delta t\theta {{\dot \gamma }^{\left( \alpha  \right)}}\left( {t + \Delta t} \right) - {{{\left\{ {\Delta {\gamma ^{\left( \alpha  \right)}}} \right\}}^{\left( k \right)}}}

        下面列出本构模型中的变量或常数与程序中变量名或常数的对应关系：

        应力 :math:`\sigma` ：stress [Pa]

        应力增量：delta_stress [Pa]

        应变 :math:`\varepsilon` ：strain [-]

        应变增量 :math:`\Delta \varepsilon` :dstrain [-]

        弹性应变增量 :math:`\Delta \varepsilon^{e}` ：delta_elastic_strain [-]

        分解剪应力 :math:`\tau` ：tau [Pa]

        位错密度 :math:`\rho` ：rho [1/m^2]

        可动位错密度 :math:`\rho_{m}` ：rho_m [1/m^2]

        位错偶极子密度 :math:`\rho_{di}` ：rho_di [1/m^2]

        热滑移阻力项 :math:`\tau_{pass}` ：tau_pass [Pa]

        滑移系滑移方向的单位向量 :math:`{\boldsymbol{m}}` 和 :math:`{\boldsymbol{m}}^{*}` ：m_s

        滑移系滑移面法向的单位向量 :math:`{\boldsymbol{n}}` 和 :math:`{\boldsymbol{n}}^{*}` ：n_s

        塑性旋率张量 :math:`{{\boldsymbol{W}}^{\rm{{p_{\text{s}}}}}}` 中的 :math:`\boldsymbol{\Omega}^{(\alpha)}` ：Omega

        塑性变形率张量 :math:`{{\boldsymbol{D}}^{\rm{{p_{\text{s}}}}}}` 中的 :math:`{\boldsymbol{P}}^{\left( \alpha  \right)}` ：P

        求解Jaumann率的中间项 :math:`{{\boldsymbol {\Omega }}^{\left( \alpha  \right)}} \cdot {\boldsymbol{\sigma}} -
        {\boldsymbol{\sigma}} \cdot {{\rm{\Omega }}^{\left( \alpha  \right)}}` ：Q

        Jaumann率中的旋转部分 :math:`\mathbb{C}:{{\boldsymbol{P}}^{\left( \alpha  \right)}} +  {{\boldsymbol {\Omega }}^{\left(
        \alpha \right)}} \cdot {\boldsymbol{\sigma}} - {\boldsymbol{\sigma}} \cdot {{\rm{\Omega }}^{\left( \alpha  \right)}}` ：S

        弹性模量张量 :math:`\mathbb{C}` ：C [Pa]

        剪切应变 :math:`\gamma` ：gamma [-]

        滑移系的剪切应变率 :math:`\dot{\gamma}` ：gamma_dot [1/s]

        剪切应变速率初值 :math:`\Delta \gamma_{t}^{(\alpha)}` ：gamma_dot_t [1/s]

        用于迭代的剪切应变速率 :math:`\Delta \gamma_{t+\Delta t}^{(\alpha)}` ：gamma_dot [1/s]

        模型中间变量 :math:`X^{\left( \alpha  \right)}` ：X [-]

        剪切模量 :math:`G` ：G [Pa]

        温度 :math:`T` ：temperature [K]

        固溶强度 :math:`\tau_{sol}` ：tau_sol [Pa]

        位错滑移速度 :math:`v_0` ：v_0 [m/s]

        平均晶粒尺寸 :math:`d` ：d_grain [m]

        玻尔兹曼常数 :math:`k_b` ：k_b [J/K]

        位错滑移阻力拟合参数 :math:`{q_{\text{s}}}` ：q_s [-]

        位错滑移阻力拟合参数 :math:`{p_{\text{s}}}` ：p_s [-]

        位错滑移激活能 :math:`Q_s` ：Q_s [J]

        位错滑移柏氏矢量长度 :math:`b_s` ：b_s [m]

        位错攀移激活能 :math:`Q_{climb}` ：Q_climb [J]

        位错湮灭拟合参数 :math:`c_{anni}` ：c_anni [-]

        平均位错间隔拟合参数 :math:`i_{slip}` ：i_slip [-]

        自扩散系数因子 :math:`D_0` ：D_0 [m^2/s]

        位错攀移激活体积 :math:`\Omega_{climb}` ：Omega_climb [m^3]

        硬化系数矩阵 :math:`H` ：H [-]

        参考文献：

        [1] A Patra, S Chaudhary, N Pai, et al., ρ-CP: Open source dislocation density based crystal plasticity framework
        for simulating temperature- and strain rate-dependent deformation, Comput. Mater. Sci., 2023, 224:112182.

        [2] 近代连续介质力学. 赵亚溥.

        [3] Nonlinear Finite Elements for Continua and Structures, Ted Belytschko.

        [4] E Orowan, Zur Kristallplastizitat. I-III, Z. für Phys., 1934, 89(9-10):605-613.

        [5] W Blum and P Eisenlohr, Dislocation mechanics of creep, Mater. Sci. Eng. A, 2009, 510-511:7–13.

        [6] A Ma and F Roters, A constitutive model for fcc single crystals based on dislocation densities and its
        application to uniaxial compression of aluminium single crystals, Acta Mater., 2004, 52(12):3603-3612.

        """
        strain = variable['strain']
        dstrain = variable['dstrain']

        if self.section.type == 'PlaneStrain':
            strain = np.array([strain[0], strain[1], 0.0, strain[2], 0.0, 0.0])
            dstrain = np.array([dstrain[0], dstrain[1], 0.0, dstrain[2], 0.0, 0.0])

        np.set_printoptions(precision=12, linewidth=256, suppress=True)

        dt = timer.dtime
        theta = self.theta
        temperature = self.temperature
        C = self.C
        G = self.G

        total_number_of_slips = self.total_number_of_slips
        tau_sol = self.tau_sol
        v_0 = self.v_0
        b_s = self.b_s
        Q_s = self.Q_s
        p_s = self.p_s
        q_s = self.q_s
        k_b = self.k_b
        d_grain = self.d_grain
        i_slip = self.i_slip
        c_anni = self.c_anni
        Q_climb = self.Q_climb
        D_0 = self.D_0
        Omega_climb = self.Omega_climb
        H = self.H
        m_s = self.m_s
        n_s = self.n_s

        d_min = c_anni * b_s

        if state_variable == {} or timer.time0 == 0.0:
            state_variable['m_s'] = m_s
            state_variable['n_s'] = n_s
            m_sxn_s = np.transpose(np.array([m_s[:, 0] * n_s[:, 0],
                                             m_s[:, 1] * n_s[:, 1],
                                             m_s[:, 2] * n_s[:, 2],
                                             2.0 * m_s[:, 0] * n_s[:, 1],
                                             2.0 * m_s[:, 0] * n_s[:, 2],
                                             2.0 * m_s[:, 1] * n_s[:, 2]]))
            n_sxm_s = np.transpose(np.array([n_s[:, 0] * m_s[:, 0],
                                             n_s[:, 1] * m_s[:, 1],
                                             n_s[:, 2] * m_s[:, 2],
                                             2.0 * n_s[:, 0] * m_s[:, 1],
                                             2.0 * n_s[:, 0] * m_s[:, 2],
                                             2.0 * n_s[:, 1] * m_s[:, 2]]))
            P = 0.5 * (m_sxn_s + n_sxm_s)
            state_variable['stress'] = np.zeros(shape=6, dtype=DTYPE)
            state_variable['tau'] = np.dot(P, state_variable['stress'])
            state_variable['gamma'] = np.zeros(shape=total_number_of_slips, dtype=DTYPE)
            state_variable['tau_pass'] = np.zeros(shape=total_number_of_slips, dtype=DTYPE)
            state_variable['rho_m'] = np.zeros(shape=total_number_of_slips, dtype=DTYPE) + 1e12
            state_variable['rho_di'] = np.zeros(shape=total_number_of_slips, dtype=DTYPE) + 1.0

        rho_m = deepcopy(state_variable['rho_m'])
        rho_di = deepcopy(state_variable['rho_di'])
        m_s = deepcopy(state_variable['m_s'])
        n_s = deepcopy(state_variable['n_s'])
        gamma = deepcopy(state_variable['gamma'])
        stress = deepcopy(state_variable['stress'])
        tau = deepcopy(state_variable['tau'])

        delta_gamma = np.zeros(shape=total_number_of_slips, dtype=DTYPE)

        is_convergence = False

        for niter in range(self.MAX_NITER):
            m_sxn_s = np.transpose(np.array([m_s[:, 0] * n_s[:, 0],
                                             m_s[:, 1] * n_s[:, 1],
                                             m_s[:, 2] * n_s[:, 2],
                                             2.0 * m_s[:, 0] * n_s[:, 1],
                                             2.0 * m_s[:, 0] * n_s[:, 2],
                                             2.0 * m_s[:, 1] * n_s[:, 2]]))

            n_sxm_s = np.transpose(np.array([n_s[:, 0] * m_s[:, 0],
                                             n_s[:, 1] * m_s[:, 1],
                                             n_s[:, 2] * m_s[:, 2],
                                             2.0 * n_s[:, 0] * m_s[:, 1],
                                             2.0 * n_s[:, 0] * m_s[:, 2],
                                             2.0 * n_s[:, 1] * m_s[:, 2]]))

            P = 0.5 * (m_sxn_s + n_sxm_s)
            Omega = 0.5 * (m_sxn_s - n_sxm_s)
            Omega[:, 3:] *= 0.5

            # S = dot(P, C) + Omega * stress - stress * Omega
            S = np.dot(P, C)

            rho = rho_di + rho_m
            tau_pass = G * b_s * np.sqrt(np.dot(H, rho))

            X = (abs(tau) - tau_pass) / tau_sol
            X_bracket = np.maximum(X, 0.0) + self.tolerance
            X_heaviside = np.sign(X_bracket)
            A_s = Q_s / k_b / temperature

            d_di = 3.0 * G * b_s / (16.0 * np.pi * (abs(tau) + self.tolerance))
            one_over_lambda = 1.0 / d_grain + 1.0 / i_slip * tau_pass / G / b_s
            v_climb = 3.0 * G * D_0 * Omega_climb / (2.0 * np.pi * k_b * temperature * (d_di + d_min)) \
                      * np.exp(-Q_climb / k_b / temperature)
            gamma_dot = rho_m * b_s * v_0 * np.exp(-A_s * (1.0 - X_bracket ** p_s) ** q_s) * np.sign(tau)

            if niter == 0:
                gamma_dot_t = deepcopy(gamma_dot)

            term1 = dt * theta
            term2 = A_s * p_s * q_s * gamma_dot * X_bracket ** (p_s - 1.0) * (1.0 - X_bracket ** p_s) ** (q_s - 1.0) \
                    * np.sign(tau)
            term3 = X_heaviside / tau_sol
            term4 = np.einsum('ik, jk->ij', S, P)
            term5 = one_over_lambda / b_s - 2.0 * d_di * rho_m / b_s
            term6 = one_over_lambda / b_s - 2.0 * d_min * rho / b_s
            term7 = 4.0 * rho_di * v_climb / (d_di - d_min) * dt
            term8 = (G * b_s) ** 2 / (2.0 * tau_pass)

            I = np.eye(total_number_of_slips, dtype=DTYPE)
            A = deepcopy(I)
            A -= term1 * term2 * term5 * b_s * v_0 * np.exp(-A_s * (1.0 - X_bracket ** p_s) ** q_s) * I
            A += term1 * term2 * term3 * term4 * np.sign(tau)
            A += term1 * term2 * term3 * term8 * np.dot(H, term6 * np.sign(tau) * I)

            if niter == 0:
                rhs = dt * gamma_dot + term1 * term2 * term3 * np.sign(tau) * np.dot(S, dstrain) \
                      + term1 * term2 * term3 * term8 * np.dot(H, term7)
                # rhs = dt * gamma_dot + term1 * term2 * term3 * sign(tau) * dot(S, dstrain)
            else:
                rhs = dt * theta * (gamma_dot - gamma_dot_t) + gamma_dot_t * dt - delta_gamma

            d_delta_gamma = np.linalg.solve(np.transpose(A), rhs)
            delta_gamma += d_delta_gamma

            delta_elastic_strain = dstrain - np.dot(delta_gamma, P)
            delta_tau = np.dot(S, delta_elastic_strain)
            delta_stress = np.dot(C, delta_elastic_strain)
            delta_rho_m = (one_over_lambda / b_s - 2.0 * d_di * rho_m / b_s) * abs(delta_gamma)
            delta_rho_di = 2.0 * (rho_m * (d_di - d_min) - rho_di * d_min) / b_s * abs(delta_gamma) - term7
            delta_m_s = 0.0
            delta_n_s = 0.0

            m_s = deepcopy(state_variable['m_s']) + delta_m_s
            n_s = deepcopy(state_variable['n_s']) + delta_n_s
            gamma = deepcopy(state_variable['gamma']) + delta_gamma
            tau = deepcopy(state_variable['tau']) + delta_tau
            stress = deepcopy(state_variable['stress']) + delta_stress
            rho_m = deepcopy(state_variable['rho_m']) + delta_rho_m
            rho_di = deepcopy(state_variable['rho_di']) + delta_rho_di

            X = (abs(tau) - tau_pass) / tau_sol
            X_bracket = np.maximum(X, 0.0)
            gamma_dot = rho_m * b_s * v_0 * np.exp(-A_s * (1.0 - X_bracket ** p_s) ** q_s) * np.sign(tau)
            residual = dt * theta * gamma_dot + dt * (1.0 - theta) * gamma_dot_t - delta_gamma

            # if element_id == 0 and iqp == 0:
            #     print('residual', residual)

            if all(residual < self.tolerance):
                is_convergence = True
                break

        ddgdde = (term1 * term2 * term3 * np.sign(tau)).reshape((total_number_of_slips, 1)) * S
        ddgdde = np.dot(np.linalg.inv(A), ddgdde)
        ddsdde = C - np.einsum('ki, kj->ij', S, ddgdde)

        if not is_convergence:
            timer.is_reduce_dtime = True

        state_variable_new['m_s'] = m_s
        state_variable_new['n_s'] = n_s
        state_variable_new['stress'] = stress
        state_variable_new['gamma'] = gamma
        state_variable_new['tau'] = tau
        state_variable_new['rho_m'] = rho_m
        state_variable_new['rho_di'] = rho_di

        if self.section.type == 'PlaneStrain':
            ddsdde = np.delete(np.delete(ddsdde, [2, 4, 5], axis=0), [2, 4, 5], axis=1)
            stress = np.delete(stress, [2, 4, 5])

        output = {'stress': stress}

        return ddsdde, output


if __name__ == "__main__":
    from pyfem.utils.visualization import print_slots_dict

    print_slots_dict(PlasticCrystalGNDs.__slots_dict__)

    from pyfem.job.Job import Job

    job = Job(r'..\..\..\examples\mechanical\1element\hex20_crystal_GNDs\Job-1.toml')

    print(job.props.materials[0].data_dict.keys())
