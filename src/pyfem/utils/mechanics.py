# -*- coding: utf-8 -*-
"""

"""
from numpy import zeros, ndarray, tensordot, array, sum, dot
from numpy.linalg import eig, inv

from pyfem.utils.colors import error_style


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


def get_transformation(v_local_0: ndarray, v_local_1: ndarray, v_local_2: ndarray,
                       v_global_0: ndarray, v_global_1: ndarray, v_global_2: ndarray) -> ndarray:
    """
    **计算空间变换矩阵**

    :param v_local_0: 局部坐标系中的向量 0
    :type v_local_0: ndarray

    :param v_local_1: 局部坐标系中的向量 1
    :type v_local_1: ndarray

    :param v_local_2: 局部坐标系中的向量 2
    :type v_local_2: ndarray

    :param v_global_0: 全局坐标系中的向量 0
    :type v_global_0: ndarray

    :param v_global_1: 全局坐标系中的向量 1
    :type v_global_1: ndarray

    :param v_global_2: 全局坐标系中的向量 2
    :type v_global_2: ndarray

    :return: 空间变换矩阵（线性）
    :rtype: ndarray

    对于空间坐标系
    """
    local_matrix = array([v_local_0, v_local_1, v_local_2])
    global_matrix = array([v_global_0, v_global_1, v_global_2])
    transformation = dot(global_matrix, inv(local_matrix))
    return transformation


def array_to_tensor_order_2(array: ndarray, dimension: int) -> ndarray:
    tensor = zeros(shape=(dimension, dimension))
    if dimension == 2:
        tensor[0, 0] = array[0]
        tensor[1, 1] = array[1]
        tensor[0, 1] = array[2]
        tensor[1, 0] = array[2]
    elif dimension == 3:
        tensor[0, 0] = array[0]
        tensor[1, 1] = array[1]
        tensor[2, 2] = array[2]
        tensor[1, 2] = array[3]
        tensor[0, 2] = array[4]
        tensor[0, 1] = array[5]
        tensor[2, 1] = array[3]
        tensor[2, 0] = array[4]
        tensor[1, 0] = array[5]
    else:
        raise NotImplementedError(error_style(f'unsupported dimension {dimension}'))
    return tensor


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
