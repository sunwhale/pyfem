# -*- coding: utf-8 -*-
"""

"""
from numpy import zeros, ndarray, tensordot, array, sum
from numpy.linalg import eig, inv

from pyfem.utils.colors import error_style


def inverse(qp_jacobis: ndarray, qp_jacobi_dets: ndarray) -> ndarray:
    """
    对于2×2和3×3的矩阵求逆直接带入下面的公式，其余的情况则调用np.linalg.inv()函数

    对于2×2的矩阵::

            | a11  a12 |
        A = |          |
            | a21  a22 |

        A^-1 = (1 / det(A)) * | a22  -a12 |
                              |           |
                              |-a21   a11 |

    对于3×3的矩阵::

            | a11  a12  a13 |
        A = |               |
            | a21  a22  a23 |
            |               |
            | a31  a32  a33 |

        A^-1 = (1 / det(A)) * |  A22*A33 - A23*A32   A13*A32 - A12*A33   A12*A23 - A13*A22 |
                              |                                                            |
                              |  A23*A31 - A21*A33   A11*A33 - A13*A31   A13*A21 - A11*A23 |
                              |                                                            |
                              |  A21*A32 - A22*A31   A12*A31 - A11*A32   A11*A22 - A12*A21 |


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
