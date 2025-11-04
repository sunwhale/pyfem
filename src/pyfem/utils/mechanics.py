# -*- coding: utf-8 -*-
"""

"""
import numpy as np

from pyfem.fem.constants import DTYPE
from pyfem.utils.colors import error_style


def inverse(qp_jacobis: np.ndarray, qp_jacobi_dets: np.ndarray) -> np.ndarray:
    r"""
    **求逆矩阵**

    :param qp_jacobis: 积分点处的雅克比矩阵列表
    :type qp_jacobis: np.ndarray

    :param qp_jacobi_dets: 积分点处的雅克比矩阵行列式列表
    :type qp_jacobi_dets: np.ndarray

    :return: 逆矩阵列表
    :rtype: np.ndarray

    当输入为 2×2 和 3×3 的矩阵直接通过解析式计算，其余的情况返回 :py:meth:`numpy.linalg.inv()` 函数的计算结果。

    对于 2×2 的矩阵：

    .. math::
        {\mathbf{A}} = \left[ {\begin{array}{*{20}{c}}
          {{A_{11}}}&{{A_{12}}} \\
          {{A_{21}}}&{{A_{22}}}
        \end{array}} \right]

    .. math::
        {{\mathbf{A}}^{ - 1}} = \frac{1}{{\det \left( {\mathbf{A}} \right)}}\left[ {\begin{array}{*{20}{c}}
          {{A_{22}}}&{ - {A_{12}}} \\
          { - {A_{21}}}&{{A_{11}}}
        \end{array}} \right]

    对于 3×3 的矩阵：

    .. math::
        {\mathbf{A}} = \left[ {\begin{array}{*{20}{c}}
          {{A_{11}}}&{{A_{12}}}&{{A_{13}}} \\
          {{A_{21}}}&{{A_{22}}}&{{A_{23}}} \\
          {{A_{31}}}&{{A_{32}}}&{{A_{33}}}
        \end{array}} \right]

    .. math::
        {{\mathbf{A}}^{ - 1}} = \frac{1}{{\det \left( {\mathbf{A}} \right)}}\left[ {\begin{array}{*{20}{c}}
          {{A_{22}}{A_{33}} - {A_{23}}{A_{32}}}&{{A_{13}}{A_{32}} - {A_{12}}{A_{33}}}&{{A_{12}}{A_{23}} - {A_{13}}{A_{22}}} \\
          {{A_{23}}{A_{31}} - {A_{21}}{A_{33}}}&{{A_{11}}{A_{33}} - {A_{13}}{A_{31}}}&{{A_{13}}{A_{21}} - {A_{11}}{A_{23}}} \\
          {{A_{21}}{A_{32}} - {A_{22}}{A_{31}}}&{{A_{12}}{A_{31}} - {A_{11}}{A_{32}}}&{{A_{11}}{A_{22}} - {A_{12}}{A_{21}}}
        \end{array}} \right]
    """

    qp_jacobi_invs = []
    for A, det_A in zip(qp_jacobis, qp_jacobi_dets):
        if A.shape == (2, 2):
            qp_jacobi_invs.append(np.array([[A[1][1], -A[0][1]], [-A[1][0], A[0][0]]]) / det_A)
        elif A.shape == (3, 3):
            qp_jacobi_invs.append(np.array([[(A[1][1] * A[2][2] - A[1][2] * A[2][1]),
                                             (A[0][2] * A[2][1] - A[0][1] * A[2][2]),
                                             (A[0][1] * A[1][2] - A[0][2] * A[1][1])],
                                            [(A[1][2] * A[2][0] - A[1][0] * A[2][2]),
                                             (A[0][0] * A[2][2] - A[0][2] * A[2][0]),
                                             (A[0][2] * A[1][0] - A[0][0] * A[1][2])],
                                            [(A[1][0] * A[2][1] - A[1][1] * A[2][0]),
                                             (A[0][1] * A[2][0] - A[0][0] * A[2][1]),
                                             (A[0][0] * A[1][1] - A[0][1] * A[1][0])]]) / det_A)
        else:
            return np.linalg.inv(qp_jacobis)
    return np.array(qp_jacobi_invs)


def get_transformation(u: np.ndarray, v: np.ndarray, w: np.ndarray,
                       u_prime: np.ndarray, v_prime: np.ndarray, w_prime: np.ndarray) -> np.ndarray:
    r"""
    **计算空间变换矩阵**

    :param u: :math:`\left( {{{{\mathbf{\hat e}}}_1},{{{\mathbf{\hat e}}}_2},{{{\mathbf{\hat e}}}_3}} \right)` 坐标系下的1号矢量
    :type u: np.ndarray

    :param v: :math:`\left( {{{{\mathbf{\hat e}}}_1},{{{\mathbf{\hat e}}}_2},{{{\mathbf{\hat e}}}_3}} \right)` 坐标系下的2号矢量
    :type v: np.ndarray

    :param w: :math:`\left( {{{{\mathbf{\hat e}}}_1},{{{\mathbf{\hat e}}}_2},{{{\mathbf{\hat e}}}_3}} \right)` 坐标系下的3号矢量
    :type w: np.ndarray

    :param u_prime: :math:`\left( {{{{\mathbf{\hat e'}}}_1},{{{\mathbf{\hat e'}}}_2},{{{\mathbf{\hat e'}}}_3}} \right)` 坐标系下的1号矢量
    :type u_prime: np.ndarray

    :param v_prime: :math:`\left( {{{{\mathbf{\hat e'}}}_1},{{{\mathbf{\hat e'}}}_2},{{{\mathbf{\hat e'}}}_3}} \right)` 坐标系下的2号矢量
    :type v_prime: np.ndarray

    :param w_prime: :math:`\left( {{{{\mathbf{\hat e'}}}_1},{{{\mathbf{\hat e'}}}_2},{{{\mathbf{\hat e'}}}_3}} \right)` 坐标系下的3号矢量
    :type w_prime: np.ndarray

    :return: 空间变换矩阵（线性）
    :rtype: np.ndarray

    设 :math:`\left( {{{{\mathbf{\hat e}}}_1},{{{\mathbf{\hat e}}}_2},{{{\mathbf{\hat e}}}_3}} \right)` 和 :math:`\left( {{{{\mathbf{\hat e'}}}_1},{{{\mathbf{\hat e'}}}_2},{{{\mathbf{\hat e'}}}_3}} \right)` 是空间 :math:`{{\mathbb{R}}^3}` 的两组基。
    如果矩阵 :math:`\mathbf{T}` 描述了两组基对应的线性变换 :math:`{{\mathbb{R}}^3} \Rightarrow {{\mathbb{R}}^3}`，则对于空间中的三个任意矢量有：

    .. math::
        \left[ {\begin{array}{*{20}{c}}
          {{T_{11}}}&{{T_{12}}}&{{T_{13}}} \\
          {{T_{21}}}&{{T_{22}}}&{{T_{23}}} \\
          {{T_{31}}}&{{T_{32}}}&{{T_{33}}}
        \end{array}} \right]\left\{ {\begin{array}{*{20}{c}}
          {{u_1}} \\
          {{u_2}} \\
          {{u_3}}
        \end{array}} \right\} = \left\{ {\begin{array}{*{20}{c}}
          {{{u'}_1}} \\
          {{{u'}_2}} \\
          {{{u'}_3}}
        \end{array}} \right\}

    .. math::
        \left[ {\begin{array}{*{20}{c}}
          {{T_{11}}}&{{T_{12}}}&{{T_{13}}} \\
          {{T_{21}}}&{{T_{22}}}&{{T_{23}}} \\
          {{T_{31}}}&{{T_{32}}}&{{T_{33}}}
        \end{array}} \right]\left\{ {\begin{array}{*{20}{c}}
          {{v_1}} \\
          {{v_2}} \\
          {{v_3}}
        \end{array}} \right\} = \left\{ {\begin{array}{*{20}{c}}
          {{{v'}_1}} \\
          {{{v'}_2}} \\
          {{{v'}_3}}
        \end{array}} \right\}

    .. math::
        \left[ {\begin{array}{*{20}{c}}
          {{T_{11}}}&{{T_{12}}}&{{T_{13}}} \\
          {{T_{21}}}&{{T_{22}}}&{{T_{23}}} \\
          {{T_{31}}}&{{T_{32}}}&{{T_{33}}}
        \end{array}} \right]\left\{ {\begin{array}{*{20}{c}}
          {{w_1}} \\
          {{w_2}} \\
          {{w_3}}
        \end{array}} \right\} = \left\{ {\begin{array}{*{20}{c}}
          {{{w'}_1}} \\
          {{{w'}_2}} \\
          {{{w'}_3}}
        \end{array}} \right\}

    合并可得：

    .. math::
        \left[ {\begin{array}{*{20}{c}}
          {{T_{11}}}&{{T_{12}}}&{{T_{13}}} \\
          {{T_{21}}}&{{T_{22}}}&{{T_{23}}} \\
          {{T_{31}}}&{{T_{32}}}&{{T_{33}}}
        \end{array}} \right]\left[ {\begin{array}{*{20}{c}}
          {{u_1}}&{{v_1}}&{{w_1}} \\
          {{u_2}}&{{v_2}}&{{w_2}} \\
          {{u_3}}&{{v_3}}&{{w_3}}
        \end{array}} \right] = \left[ {\begin{array}{*{20}{c}}
          {{{u'}_1}}&{{{v'}_1}}&{{{w'}_1}} \\
          {{{u'}_2}}&{{{v'}_2}}&{{{w'}_2}} \\
          {{{u'}_3}}&{{{v'}_3}}&{{{w'}_3}}
        \end{array}} \right]

    因此变换矩阵 :math:`\mathbf{T}` 可以表示为：

    .. math::
        \left[ {\begin{array}{*{20}{c}}
          {{T_{11}}}&{{T_{12}}}&{{T_{13}}} \\
          {{T_{21}}}&{{T_{22}}}&{{T_{23}}} \\
          {{T_{31}}}&{{T_{32}}}&{{T_{33}}}
        \end{array}} \right] = \left[ {\begin{array}{*{20}{c}}
          {{{u'}_1}}&{{{v'}_1}}&{{{w'}_1}} \\
          {{{u'}_2}}&{{{v'}_2}}&{{{w'}_2}} \\
          {{{u'}_3}}&{{{v'}_3}}&{{{w'}_3}}
        \end{array}} \right]{\left[ {\begin{array}{*{20}{c}}
          {{u_1}}&{{v_1}}&{{w_1}} \\
          {{u_2}}&{{v_2}}&{{w_2}} \\
          {{u_3}}&{{v_3}}&{{w_3}}
        \end{array}} \right]^{ - 1}}

    记为：

    .. math::
        {\mathbf{T}} = {\mathbf{A'}} \cdot {\left( {\mathbf{A}} \right)^{ - 1}}

    特殊的，对于两个空间直角坐标系的映射，即 :math:`\left( {{{{\mathbf{\hat e}}}_1},{{{\mathbf{\hat e}}}_2},{{{\mathbf{\hat e}}}_3}} \right)` 和 :math:`\left( {{{{\mathbf{\hat e'}}}_1},{{{\mathbf{\hat e'}}}_2},{{{\mathbf{\hat e'}}}_3}} \right)` 均为单位正交基，且变换需要满足  :math:`{{\mathbb{R}}^3} \Rightarrow {{\mathbb{R}}^3}` ，则 :math:`{\mathbf{A}}` 和  :math:`{{\mathbf{A'}}}` 必须满秩且为正交矩阵。
    """

    A = np.array([u, v, w])
    A_prime = np.array([u_prime, v_prime, w_prime])
    T = np.dot(A_prime, np.linalg.inv(A))
    return T


def voigt_array_to_tensor(voigt_array: np.ndarray, dimension: int) -> np.ndarray:
    r"""
    **Voigt记法数组转换为2阶张量**

    :param voigt_array: Voigt记法数组
    :type voigt_array: np.ndarray

    :param dimension: 空间维度
    :type dimension: int

    :return: 2阶张量
    :rtype: np.ndarray

    映射方式：

    .. math::
        \left\{ {\begin{array}{*{20}{c}}
          {{T_{11}}} \\
          {{T_{22}}} \\
          {{T_{12}}}
        \end{array}} \right\} \to \left[ {\begin{array}{*{20}{c}}
          {{T_{11}}}&{{T_{12}}}} \\
          {{T_{21}}}&{{T_{22}}}
        \end{array}} \right]

    .. math::
        \left\{ {\begin{array}{*{20}{c}}
          {{T_{11}}} \\
          {{T_{22}}} \\
          {{T_{33}}} \\
          {{T_{12}}} \\
          {{T_{13}}} \\
          {{T_{23}}}
        \end{array}} \right\} \to \left[ {\begin{array}{*{20}{c}}
          {{T_{11}}}&{{T_{12}}}&{{T_{13}}} \\
          {{T_{21}}}&{{T_{22}}}&{{T_{23}}} \\
          {{T_{31}}}&{{T_{32}}}&{{T_{33}}}
        \end{array}} \right]
    """

    tensor = np.zeros(shape=(dimension, dimension))
    if dimension == 2:
        tensor[0, 0] = voigt_array[0]
        tensor[1, 1] = voigt_array[1]
        tensor[0, 1] = voigt_array[2]
        tensor[1, 0] = voigt_array[2]
    elif dimension == 3:
        tensor[0, 0] = voigt_array[0]
        tensor[1, 1] = voigt_array[1]
        tensor[2, 2] = voigt_array[2]
        tensor[0, 1] = voigt_array[3]
        tensor[0, 2] = voigt_array[4]
        tensor[1, 2] = voigt_array[5]
        tensor[1, 0] = voigt_array[3]
        tensor[2, 0] = voigt_array[4]
        tensor[2, 1] = voigt_array[5]
    else:
        raise NotImplementedError(error_style(f'unsupported dimension {dimension}'))
    return tensor


def get_voigt_transformation(transformation: np.ndarray) -> np.ndarray:
    """
    **获取旋转矩阵的Voigt形式**

    :param transformation: Voigt记法数组
    :type transformation: np.ndarray

    :return: voigt_transformation
    :rtype: np.ndarray
    """

    voigt_transformation = np.zeros(shape=(6, 6), dtype=DTYPE)

    a11 = transformation[0][0]
    a12 = transformation[0][1]
    a13 = transformation[0][2]
    a21 = transformation[1][0]
    a22 = transformation[1][1]
    a23 = transformation[1][2]
    a31 = transformation[2][0]
    a32 = transformation[2][1]
    a33 = transformation[2][2]

    voigt_transformation[0][0] = a11 ** 2
    voigt_transformation[0][1] = a12 ** 2
    voigt_transformation[0][2] = a13 ** 2
    voigt_transformation[0][3] = 2 * a11 * a12
    voigt_transformation[0][4] = 2 * a11 * a13
    voigt_transformation[0][5] = 2 * a12 * a13

    voigt_transformation[1][0] = a21 ** 2
    voigt_transformation[1][1] = a22 ** 2
    voigt_transformation[1][2] = a23 ** 2
    voigt_transformation[1][3] = 2 * a21 * a22
    voigt_transformation[1][4] = 2 * a21 * a23
    voigt_transformation[1][5] = 2 * a22 * a23

    voigt_transformation[2][0] = a31 ** 2
    voigt_transformation[2][1] = a32 ** 2
    voigt_transformation[2][2] = a33 ** 2
    voigt_transformation[2][3] = 2 * a31 * a32
    voigt_transformation[2][4] = 2 * a31 * a33
    voigt_transformation[2][5] = 2 * a32 * a33

    voigt_transformation[3][0] = a11 * a21
    voigt_transformation[3][1] = a12 * a22
    voigt_transformation[3][2] = a13 * a23
    voigt_transformation[3][3] = a12 * a21 + a11 * a22
    voigt_transformation[3][4] = a13 * a21 + a11 * a23
    voigt_transformation[3][5] = a13 * a22 + a12 * a23

    voigt_transformation[4][0] = a11 * a31
    voigt_transformation[4][1] = a12 * a32
    voigt_transformation[4][2] = a13 * a33
    voigt_transformation[4][3] = a12 * a31 + a11 * a32
    voigt_transformation[4][4] = a13 * a31 + a11 * a33
    voigt_transformation[4][5] = a13 * a32 + a12 * a33

    voigt_transformation[5][0] = a21 * a31
    voigt_transformation[5][1] = a22 * a32
    voigt_transformation[5][2] = a23 * a33
    voigt_transformation[5][3] = a22 * a31 + a21 * a32
    voigt_transformation[5][4] = a23 * a31 + a21 * a33
    voigt_transformation[5][5] = a23 * a32 + a22 * a33

    return voigt_transformation


# def get_decompose_energy(strain: np.ndarray, solid_material_data0: float, solid_material_data1: float, dimension: int):
def get_decompose_energy(strain: np.ndarray, stress: np.ndarray, dimension: int):
    r"""
    **获取正的应变能密度**

    定义一种基于应变能谱分解的应变能分解方式（miehe）分解方法

    :param strain: Voigt记法数组
    :type strain: np.ndarray

    :param stress: Voigt记法数组
    :type stress: np.ndarray

    :param dimension: 单元维度
    :type dimension: int

    :return: energy_positive
    :rtype: np.ndarray

    :return: energy_negative
    :rtype: np.ndarray

    对于相场演化方程，如何选取合适的驱动力是尤为重要的。下面介绍一种通过能量分解的方法，在应变能中发掘驱动材料损伤的分量。

    2010年，miehe等[1]给出了一种基于应变张量谱分解的应变能分解方式，使用这种分解方式，可以得到一种拉压破坏模式不一致的相场模型，并且能够防止裂纹面的相互侵入问题。

    应变张量的谱分解表示为：

    .. math::
        {{\mathbf{\varepsilon}} }{\text{ = }}\sum\limits_{i = 1}^3 { \varepsilon _i } {n_i } \otimes {n_i }

    其中， :math:`{\varepsilon _i}(i = 1,2,3)` 为三个主应变分量。一个变形张量可以分解为三个不包含剪切变形，只拉伸或压缩变形的状态，这三个方向称为主应变方向，对应的大小为主应变。
    主应变及其方向可以通过求解应变矩阵的特征根来获得。下面求解应变张量不变量与主应变分量[2]。定义一个二阶应变张量为：

    .. math::
        {\mathbf{\varepsilon }} = \left[ {\begin{array}{*{20}{c}}
          {{\varepsilon _{11}}}&{{\varepsilon _{12}}}&{{\varepsilon _{13}}} \\
          {{\varepsilon _{21}}}&{{\varepsilon _{22}}}&{{\varepsilon _{23}}} \\
          {{\varepsilon _{31}}}&{{\varepsilon _{32}}}&{{\varepsilon _{33}}}
        \end{array}} \right]

    令主应变为 :math:`{\varepsilon _e}(e = 1,2,3)` ，则 :math:`{\varepsilon _e}` 可由下列方程求得

    .. math::
        \left| {\begin{array}{*{20}{c}}
          {{\varepsilon _{11}} - {\varepsilon _e}}&{{\varepsilon _{12}}}&{{\varepsilon _{13}}} \\
          {{\varepsilon _{21}}}&{{\varepsilon _{22}} - {\varepsilon _e}}&{{\varepsilon _{23}}} \\
          {{\varepsilon _{31}}}&{{\varepsilon _{32}}}&{{\varepsilon _{33}} - {\varepsilon _e}}
        \end{array}} \right| = 0

    上式可写为如下形式：

    .. math::
        \varepsilon _e^3 - {J_1}\varepsilon _e^2 + {J_2}{\varepsilon _e} - {J_3} = 0

    其中，方程的三个根即为三个主应变 :math:`{\varepsilon _1}，{\varepsilon _2}，{\varepsilon _3}` 。通常按照 :math:`{\varepsilon _1} \geqslant {\varepsilon _2} \geqslant {\varepsilon _3}` 来排列。
    系数项 :math:`{J_1}, {J_2}, {J_3}` 分别称为应变张量的第一、第二、第三不变量。

    其中，第一不变量 :math:`{J_1}` 定义为应变张量的迹 :math:`{J_1} = trace({\mathbf{\varepsilon }})` ;

    .. math::
        {J_1} = {\varepsilon _{11}} + {\varepsilon _{22}} + {\varepsilon _{33}} = {\varepsilon _1} +
        {\varepsilon _2} + {\varepsilon _3} = trace({\mathbf{\varepsilon }})

    第二不变量 :math:`{J_2}` 通常称为等效应变， :math:`{J_2} = \frac{1}{2}\left[ {{{\left( {trace\left( {\mathbf{\varepsilon }} \right)} \right)}^2} - trace\left( {{{\mathbf{\varepsilon }}^2}} \right)} \right]` ;

    .. math::
        {J_2} = \left| {\begin{array}{*{20}{c}}
          {{\varepsilon _{22}}}&{{\varepsilon _{23}}} \\
          {{\varepsilon _{32}}}&{{\varepsilon _{33}}}
        \end{array}} \right| + \left| {\begin{array}{*{20}{c}}
          {{\varepsilon _{11}}}&{{\varepsilon _{13}}} \\
          {{\varepsilon _{31}}}&{{\varepsilon _{33}}}
        \end{array}} \right| + \left| {\begin{array}{*{20}{c}}
          {{\varepsilon _{11}}}&{{\varepsilon _{12}}} \\
          {{\varepsilon _{21}}}&{{\varepsilon _{22}}}
        \end{array}} \right| =
        {\varepsilon _1}{\varepsilon _2} + {\varepsilon _2}{\varepsilon _3} + {\varepsilon _3}{\varepsilon _1} =
        \frac{1}{2}\left[ {{{\left( {trace\left( {\mathbf{\varepsilon }} \right)} \right)}^2} - trace\left( {{{\mathbf{\varepsilon }}^2}} \right)} \right]

    第三不变量 :math:`{J_3}` 定义为矩阵行列式， :math:`{J_3} = \det ({\mathbf{\varepsilon }})` 。

    .. math::
        {J_3} = \left| {\begin{array}{*{20}{c}}
          {{\varepsilon _{11}}}&{{\varepsilon _{12}}}&{{\varepsilon _{13}}} \\
          {{\varepsilon _{21}}}&{{\varepsilon _{22}}}&{{\varepsilon _{23}}} \\
          {{\varepsilon _{31}}}&{{\varepsilon _{32}}}&{{\varepsilon _{33}}}
        \end{array}} \right| = {\varepsilon _1}{\varepsilon _2}{\varepsilon _3} = \det ({\mathbf{\varepsilon }})

    用主应变分量可将应变能密度函数表示为：

    .. math::
        \psi ({\mathbf{\varepsilon }}) = \frac{1}{2}\lambda {\left( {{\varepsilon _1} + {\varepsilon _2} +
        {\varepsilon _3}} \right)^2} + \mu \left( {\varepsilon _1^2 + \varepsilon _2^2 + \varepsilon _3^2} \right)

    将上面的名义应变能密度函数分解为受拉和受压两个部分：

    .. math::
        \psi  = {\psi ^ + } + {\psi ^ - }

    对于各向同性线弹性材料，应变能密度函数可用应变张量表示为：

    .. math::
        {\psi ^ \pm }({\mathbf{\varepsilon }}) = \frac{1}{2}\lambda \left\langle {trace\left( {\mathbf{\varepsilon }} \right)} \right\rangle _ \pm ^2 +
        \mu {\text{ }}trace\left( {\left\langle {\varepsilon } \right\rangle _ \pm ^2} \right)

    即：

    .. math::
        {\psi ^ \pm }({\mathbf{\varepsilon }}) = \frac{1}{2}\lambda \left\langle {{\varepsilon _1} + {\varepsilon _2} + {\varepsilon _3}} \right\rangle _ \pm ^2 +
        \mu \left( {\left\langle {{\varepsilon _1}} \right\rangle _ \pm ^2 + \left\langle {{\varepsilon _2}} \right\rangle _ \pm ^2 + \left\langle {{\varepsilon _3}} \right\rangle _ \pm ^2} \right)

    其中， :math:`{\left\langle  \cdot  \right\rangle _ \pm }` 为Macaulay括号，表示为：

    .. math::
        {\left\langle x \right\rangle _ + }: = \left( {\left| x \right| + x} \right)/2;
        {\left\langle x \right\rangle _ - }: = \left( {\left| x \right| - x} \right)/2

    图像描述为::

                    <x>_{+}                                                <x>_{-}
                       |                                                      |
                       |                *              *                      |
                       |              *                  *                    |
                       |            *                       *                 |
                       |          *                            *              |
                       |        *                                 *           |
                       |      *                                      *        |
                       |    *                                           *     |
                       |  *                                                *  |
        ----***********0--x-------------------->   ---------------------------0**x**************-------->

    在分解后的应变能密度中，假设只有受拉应变能  :math:`{\psi ^ + }`  驱动相场的演化。而受压部分并不引起相场的演化，于是相场的驱动方程

    .. math::
        {F_d} =  - {w^{'}}(d) {\psi }( \varepsilon )

    可以改写为：

    .. math::
        {F_d} =  - {w^{'}}(d) {\psi ^ + }( \varepsilon )

    参考文献：

    [1] miehe, et.al., Thermodynamically consistent phase-field models of fracture: Variational principles and multi-field FE implementations. 2010.
    https://onlinelibrary.wiley.com/doi/abs/10.1002/nme.2861

    [2] 固体力学 [尹祥础 编] 2011年版

    """
    E = 1.0e5
    nu = 0.25
    mu = E / (2.0 * (1.0 + nu))
    lame = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))

    strain = voigt_array_to_tensor(strain, dimension)

    # 得到主应变张量分量与特征方向
    principle_strain_value, principle_strain_vector = np.linalg.eig(strain)
    # print('principle_strain_value', principle_strain_value)

    # 1. 用向量乘法，计算高效，语法简洁
    # strain_positive = zeros(shape=(dimension, dimension))
    # strain_negative = zeros(shape=(dimension, dimension))
    #
    # for i in range(dimension):
    #     strain_positive += 0.5 * (principle_strain_value[i] + abs(principle_strain_value[i])) * \
    #                        tensordot(principle_strain_vector[:, i], principle_strain_vector[:, i], 0)
    #     strain_negative += 0.5 * (principle_strain_value[i] - abs(principle_strain_value[i])) * \
    #                        tensordot(principle_strain_vector[:, i], principle_strain_vector[:, i], 0)
    #     # tensordot(a,b,axes=1); 表示取a的最后几个维度,与b的前面几个维度相乘,再累加求和
    #     # 如： axes=1是指取a矩阵的最后一个维度，与b矩阵的第一个维度相乘
    #
    # energy_positive = 0.5 * lame * (0.5 * (strain.trace() + abs(strain.trace()))) ** 2 + \
    #                   mu * (strain_positive * strain_positive).trace()
    #
    # energy_negative = 0.5 * lame * (0.5 * (abs(strain.trace()) - strain.trace())) ** 2 + \
    #                   mu * (strain_negative * strain_negative).trace()
    #
    # print('energy_positive', energy_positive)
    # print('energy_negative', energy_negative)

    # energy = 0.5 * lame * (0.5 * strain.trace()) ** 2 + mu * (strain * strain).trace()

    # 2. 用函数定义，便于理解
    # 得到 <x>_{+} 与 <x>_{-}
    energy_positive = 0.0
    energy_negative = 0.0
    if dimension == 2:
        ep1 = max(principle_strain_value)
        ep2 = min(principle_strain_value)
        eq_sum = sum(principle_strain_value)
        if ep1 > 0.0:
            ep1_p = (abs(ep1) + ep1) / 2.0
            ep1_n = 0.0
        else:
            ep1_p = 0.0
            ep1_n = (abs(ep1) - ep1) / 2.0
        if ep2 > 0.0:
            ep2_p = (abs(ep2) + ep2) / 2.0
            ep2_n = 0.0
        else:
            ep2_p = 0.0
            ep2_n = (abs(ep2) - ep2) / 2.0
        if eq_sum > 0.0:
            eq_sum_p = (abs(eq_sum) + eq_sum) / 2.0
            eq_sum_n = 0.0
        else:
            eq_sum_p = 0.0
            eq_sum_n = (abs(eq_sum) - eq_sum) / 2.0
        energy_positive = lame / 2.0 * eq_sum_p ** 2.0 + mu * (ep1_p ** 2.0 + ep2_p ** 2.0)
        energy_negative = lame / 2.0 * eq_sum_n ** 2.0 + mu * (ep1_n ** 2.0 + ep2_n ** 2.0)

    elif dimension == 3:
        ep1 = max(principle_strain_value)
        ep3 = min(principle_strain_value)
        eq_sum = sum(principle_strain_value)
        ep2 = eq_sum - ep1 - ep3
        if ep1 > 0.0:
            ep1_p = (abs(ep1) + ep1) / 2.0
            ep1_n = 0.0
        else:
            ep1_p = 0.0
            ep1_n = (abs(ep1) - ep1) / 2.0
        if ep2 > 0.0:
            ep2_p = (abs(ep2) + ep2) / 2.0
            ep2_n = 0.0
        else:
            ep2_p = 0.0
            ep2_n = (abs(ep2) - ep2) / 2.0
        if ep3 > 0.0:
            ep3_p = (abs(ep3) + ep3) / 2.0
            ep3_n = 0.0
        else:
            ep3_p = 0.0
            ep3_n = (abs(ep3) - ep3) / 2.0
        if eq_sum > 0.0:
            eq_sum_p = (abs(eq_sum) + eq_sum) / 2.0
            eq_sum_n = 0.0
        else:
            eq_sum_p = 0.0
            eq_sum_n = (abs(eq_sum) - eq_sum) / 2.0
        energy_positive = lame / 2.0 * eq_sum_p ** 2.0 + mu * (ep1_p ** 2.0 + ep2_p ** 2.0 + ep3_p ** 2.0)
        energy_negative = lame / 2.0 * eq_sum_n ** 2.0 + mu * (ep1_n ** 2.0 + ep2_n ** 2.0 + ep3_n ** 2.0)

    # print('energy_positive', energy_positive)
    # print('energy_negative', energy_negative)

    return energy_positive, energy_negative


def operations_for_symtensor_antisymtensor(sym_tensor: np.ndarray, antisym_tensor: np.ndarray) -> np.ndarray:
    r"""
    **获取反对称张量乘以对称张量减去对称张量乘以反对称张量的结果**

    :param sym_tensor: :math:`\left[ {{a_{11}}{\text{ }}{a_{22}}{\text{ }}{a_{33}}{\text{ }}{a_{12}}{\text{ }}{a_{13}}{\text{ }}{a_{23}}} \right]` 采用 Vogit 记法的对称张量数组
    :type sym_tensor: np.ndarray

    :param antisym_tensor: :math:`\left[ {0{\text{ }}0{\text{ }}0{\text{ }}{b_{12}}{\text{ }}{b_{13}}{\text{ }}{b_{23}}} \right]` 采用类似 Vogit 记法的反对称张量数组
    :type antisym_tensor: np.ndarray

    :return: 反对称张量乘以对称张量减去对称张量乘以反对称张量的结果
    :rtype: np.ndarray

    首先分别将对称张量数组 :math:`\left[ {{a_{11}}{\text{ }}{a_{22}}{\text{ }}{a_{33}}{\text{ }}{a_{12}}{\text{ }}{a_{13}}{\text{ }}{a_{23}}} \right]` 和
    反对称张量数组 :math:`\left[ {0{\text{ }}0{\text{ }}0{\text{ }}{b_{12}}{\text{ }}{b_{13}}{\text{ }}{b_{23}}} \right]` 写成矩阵形式：

    .. math::
        A_{sym} = \left[ {\begin{array}{*{20}{c}}
          {{a_{11}}}&{{a_{12}}}&{{a_{13}}} \\
          {{a_{12}}}&{{a_{22}}}&{{a_{23}}} \\
          {{a_{13}}}&{{a_{23}}}&{{a_{33}}}
        \end{array}} \right],B_{antisym} = \left[ {\begin{array}{*{20}{c}}
          0&{{b_{12}}}&{{b_{13}}} \\
          { - {b_{12}}}&0&{{b_{23}}} \\
          { - {b_{13}}}&{ - {b_{23}}}&0
        \end{array}} \right]

    计算结果 :math:`{B_{antisym}}{A_{sym}} - {A_{sym}}{B_{antisym}}` 表示为：

    .. math::
        \begin{gathered}
          {B_{antisym}}{A_{sym}} - {A_{sym}}{B_{antisym}} = \left[ {\begin{array}{*{20}{c}}
          0&0&0&{2{b_{12}}}&{2{b_{13}}}&0 \\
          0&0&0&{ - 2{b_{12}}}&0&{2{b_{23}}} \\
          0&0&0&0&{ - 2{b_{13}}}&{ - 2{b_{23}}} \\
          { - {b_{12}}}&{{b_{12}}}&0&0&{{b_{23}}}&{{b_{13}}} \\
          { - {b_{13}}}&0&{{b_{13}}}&{ - {b_{23}}}&0&{{b_{12}}} \\
          0&{ - {b_{23}}}&{{b_{23}}}&{ - {b_{13}}}&{ - {b_{12}}}&0
        \end{array}} \right]\left[ \begin{gathered}
          {a_{11}} \hfill \\
          {a_{22}} \hfill \\
          {a_{33}} \hfill \\
          {a_{12}} \hfill \\
          {a_{13}} \hfill \\
          {a_{23}} \hfill \\
        \end{gathered}  \right] \hfill \\
           = \left[ \begin{gathered}
          2{b_{12}}{a_{12}} + 2{b_{13}}{a_{13}} \hfill \\
           - 2{b_{12}}{a_{12}} + 2{b_{23}}{a_{23}} \hfill \\
           - 2{b_{13}}{a_{13}} - 2{b_{23}}{a_{23}} \hfill \\
           - {b_{12}}{a_{11}} + {b_{12}}{a_{22}} + {b_{23}}{a_{13}} + {b_{13}}{a_{23}} \hfill \\
           - {b_{13}}{a_{11}} + {b_{13}}{a_{33}} - {b_{23}}{a_{12}} + {b_{12}}{a_{23}} \hfill \\
           - {b_{23}}{a_{22}} + {b_{23}}{a_{33}} - {b_{13}}{a_{12}} - {b_{12}}{a_{13}} \hfill \\
        \end{gathered}  \right] = T{A_{sym}} \hfill \\
        \end{gathered}
    """
    result = np.zeros(shape=(len(antisym_tensor), 6), dtype=DTYPE)

    for i in range(len(antisym_tensor)):
        T = np.array([[0, 0, 0, 2 * antisym_tensor[i, 3], 2 * antisym_tensor[i, 4], 0],
                      [0, 0, 0, -2 * antisym_tensor[i, 3], 0, 2 * antisym_tensor[i, 5]],
                      [0, 0, 0, 0, -2 * antisym_tensor[i, 4], -2 * antisym_tensor[i, 5]],
                      [-antisym_tensor[i, 3], antisym_tensor[i, 3], 0, 0, antisym_tensor[i, 5], antisym_tensor[i, 4]],
                      [-antisym_tensor[i, 4], 0, antisym_tensor[i, 4], -antisym_tensor[i, 5], 0, antisym_tensor[i, 3]],
                      [0, -antisym_tensor[i, 5], antisym_tensor[i, 5], -antisym_tensor[i, 4], -antisym_tensor[i, 3], 0]])
        result[i, :] = np.dot(T, sym_tensor)

    return result


if __name__ == '__main__':
    # u = array([1, 0, 0])
    # v = array([0, 1, 0])
    # w = array([0, 0, 1])
    #
    # u_prime = array([1, 0, 0])
    # v_prime = array([0, 1, 0])
    # w_prime = array([0, 0, 0.6187])
    # T = get_transformation(u, v, w, u_prime, v_prime, w_prime)
    #
    # n = array([1, 0, 0])
    # n = dot(T, n)
    # n *= 1.0 / norm(n)
    #
    # print(n)
    strain = np.array([-0.7, 0.6, 0.12])
    stress = np.array([-500, 643, 120])
    a, b = get_decompose_energy(strain, stress, 2)
    # print(a, b)
