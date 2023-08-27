# -*- coding: utf-8 -*-
"""

"""
from numpy import zeros, ndarray, array, sum, dot
from numpy.linalg import inv, norm

from pyfem.utils.colors import error_style
from pyfem.fem.constants import DTYPE


def inverse(qp_jacobis: ndarray, qp_jacobi_dets: ndarray) -> ndarray:
    r"""
    **求逆矩阵**

    :param qp_jacobis: 积分点处的雅克比矩阵列表
    :type qp_jacobis: ndarray

    :param qp_jacobi_dets: 积分点处的雅克比矩阵行列式列表
    :type qp_jacobi_dets: ndarray

    :return: 逆矩阵列表
    :rtype: ndarray

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
            qp_jacobi_invs.append(array([[A[1][1], -A[0][1]], [-A[1][0], A[0][0]]]) / det_A)
        elif A.shape == (3, 3):
            qp_jacobi_invs.append(array([[(A[1][1] * A[2][2] - A[1][2] * A[2][1]),
                                          (A[0][2] * A[2][1] - A[0][1] * A[2][2]),
                                          (A[0][1] * A[1][2] - A[0][2] * A[1][1])],
                                         [(A[1][2] * A[2][0] - A[1][0] * A[2][2]),
                                          (A[0][0] * A[2][2] - A[0][2] * A[2][0]),
                                          (A[0][2] * A[1][0] - A[0][0] * A[1][2])],
                                         [(A[1][0] * A[2][1] - A[1][1] * A[2][0]),
                                          (A[0][1] * A[2][0] - A[0][0] * A[2][1]),
                                          (A[0][0] * A[1][1] - A[0][1] * A[1][0])]]) / det_A)
        else:
            return inv(qp_jacobis)
    return array(qp_jacobi_invs)


def get_transformation(u: ndarray, v: ndarray, w: ndarray,
                       u_prime: ndarray, v_prime: ndarray, w_prime: ndarray) -> ndarray:
    r"""
    **计算空间变换矩阵**

    :param u: :math:`\left( {{{{\mathbf{\hat e}}}_1},{{{\mathbf{\hat e}}}_2},{{{\mathbf{\hat e}}}_3}} \right)` 坐标系下的1号矢量
    :type u: ndarray

    :param v: :math:`\left( {{{{\mathbf{\hat e}}}_1},{{{\mathbf{\hat e}}}_2},{{{\mathbf{\hat e}}}_3}} \right)` 坐标系下的2号矢量
    :type v: ndarray

    :param w: :math:`\left( {{{{\mathbf{\hat e}}}_1},{{{\mathbf{\hat e}}}_2},{{{\mathbf{\hat e}}}_3}} \right)` 坐标系下的3号矢量
    :type w: ndarray

    :param u_prime: :math:`\left( {{{{\mathbf{\hat e'}}}_1},{{{\mathbf{\hat e'}}}_2},{{{\mathbf{\hat e'}}}_3}} \right)` 坐标系下的1号矢量
    :type u_prime: ndarray

    :param v_prime: :math:`\left( {{{{\mathbf{\hat e'}}}_1},{{{\mathbf{\hat e'}}}_2},{{{\mathbf{\hat e'}}}_3}} \right)` 坐标系下的2号矢量
    :type v_prime: ndarray

    :param w_prime: :math:`\left( {{{{\mathbf{\hat e'}}}_1},{{{\mathbf{\hat e'}}}_2},{{{\mathbf{\hat e'}}}_3}} \right)` 坐标系下的3号矢量
    :type w_prime: ndarray

    :return: 空间变换矩阵（线性）
    :rtype: ndarray

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

    A = array([u, v, w])
    A_prime = array([u_prime, v_prime, w_prime])
    T = dot(A_prime, inv(A))
    return T


def vogit_array_to_tensor(vogit_array: ndarray, dimension: int) -> ndarray:
    r"""
    **Voigt记法数组转换为2阶张量**

    :param vogit_array: Voigt记法数组
    :type vogit_array: ndarray

    :param dimension: 空间维度
    :type dimension: int

    :return: 2阶张量
    :rtype: ndarray

    映射方式：

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

    tensor = zeros(shape=(dimension, dimension))
    if dimension == 2:
        tensor[0, 0] = vogit_array[0]
        tensor[1, 1] = vogit_array[1]
        tensor[0, 1] = vogit_array[2]
        tensor[1, 0] = vogit_array[2]
    elif dimension == 3:
        tensor[0, 0] = vogit_array[0]
        tensor[1, 1] = vogit_array[1]
        tensor[2, 2] = vogit_array[2]
        tensor[0, 1] = vogit_array[3]
        tensor[0, 2] = vogit_array[4]
        tensor[1, 2] = vogit_array[5]
        tensor[1, 0] = vogit_array[3]
        tensor[2, 0] = vogit_array[4]
        tensor[2, 1] = vogit_array[5]
    else:
        raise NotImplementedError(error_style(f'unsupported dimension {dimension}'))
    return tensor


def get_voigt_transformation(transformation: ndarray) -> ndarray:
    """
    **获取旋转矩阵的Voigt形式**

    :param transformation: Voigt记法数组
    :type transformation: ndarray

    :return: voigt_transformation
    :rtype: ndarray
    """

    voigt_transformation = zeros(shape=(6, 6), dtype=DTYPE)

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


def get_decompose_energy(strain: ndarray, stress: ndarray, dimension: int):
    # strain = array_to_tensor_order_2(strain, dimension)
    # # stress = array_to_tensor_order_2(stress, dimension)
    #
    # principle_strain_value, principle_strain_vector = eig(strain)
    # # principle_stress_value, principle_stress_vector = eig(stress)
    #
    # strain_positive = zeros(shape=(dimension, dimension))
    # strain_negative = zeros(shape=(dimension, dimension))
    #
    # # stress_positive = zeros(shape=(dimension, dimension))
    # # stress_negative = zeros(shape=(dimension, dimension))
    #
    # for i in range(dimension):
    #     strain_positive += 0.5 * (principle_strain_value[i] + abs(principle_strain_value[i])) * \
    #                        tensordot(principle_strain_vector[:, i], principle_strain_vector[:, i], 0)
    #
    #     strain_negative += 0.5 * (principle_strain_value[i] - abs(principle_strain_value[i])) * \
    #                        tensordot(principle_strain_vector[:, i], principle_strain_vector[:, i], 0)
    #
    #     # stress_positive += 0.5 * (principle_stress_value[i] + abs(principle_stress_value[i])) * \
    #     #                    tensordot(principle_stress_vector[:, i], principle_stress_vector[:, i], 0)
    #     #
    #     # stress_negative += 0.5 * (principle_stress_value[i] - abs(principle_stress_value[i])) * \
    #     #                    tensordot(principle_stress_vector[:, i], principle_stress_vector[:, i], 0)
    #
    # E = 1.0e5
    # nu = 0.25
    #
    # mu = E / (2 * (1 + nu))
    # lame = (E * nu) / ((1 + nu) * (1 - 2 * nu))
    #
    # energy_positive = 0.5 * lame * (0.5 * (strain.trace() + abs(strain.trace()))) ** 2 + \
    #                   mu * (strain_positive * strain_positive).trace()
    #
    # energy_negative = 0.5 * lame * (0.5 * (strain.trace() - abs(strain.trace()))) ** 2 + \
    #                   mu * (strain_negative * strain_negative).trace()

    energy_positive = 0.5 * sum(stress * strain)

    return energy_positive, energy_positive


if __name__ == '__main__':
    u = array([1, 0, 0])
    v = array([0, 1, 0])
    w = array([0, 0, 1])

    u_prime = array([1, 0, 0])
    v_prime = array([0, 1, 0])
    w_prime = array([0, 0, 0.6187])
    T = get_transformation(u, v, w, u_prime, v_prime, w_prime)

    n = array([1, 0, 0])
    n = dot(T, n)
    n *= 1.0 / norm(n)

    print(n)