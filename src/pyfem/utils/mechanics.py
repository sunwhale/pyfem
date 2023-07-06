# -*- coding: utf-8 -*-
"""

"""
from numpy import zeros, ndarray, tensordot, dot
from numpy.linalg import eig

from pyfem.utils.colors import error_style


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

    strain = array_to_tensor_order_2(strain, dimension)
    # stress = array_to_tensor_order_2(stress, dimension)

    principle_strain_value, principle_strain_vector = eig(strain)
    # principle_stress_value, principle_stress_vector = eig(stress)

    strain_positive = zeros(shape=(dimension, dimension))
    strain_negative = zeros(shape=(dimension, dimension))

    # stress_positive = zeros(shape=(dimension, dimension))
    # stress_negative = zeros(shape=(dimension, dimension))

    for i in range(dimension):
        strain_positive += 0.5 * (principle_strain_value[i] + abs(principle_strain_value[i])) * \
                           tensordot(principle_strain_vector[:, i], principle_strain_vector[:, i], 0)

        strain_negative += 0.5 * (principle_strain_value[i] - abs(principle_strain_value[i])) * \
                           tensordot(principle_strain_vector[:, i], principle_strain_vector[:, i], 0)

        # stress_positive += 0.5 * (principle_stress_value[i] + abs(principle_stress_value[i])) * \
        #                    tensordot(principle_stress_vector[:, i], principle_stress_vector[:, i], 0)
        #
        # stress_negative += 0.5 * (principle_stress_value[i] - abs(principle_stress_value[i])) * \
        #                    tensordot(principle_stress_vector[:, i], principle_stress_vector[:, i], 0)

    E = 1.0e5
    nu = 0.25

    mu = E / (2 * (1 + nu))
    lame = (E * nu) / ((1 + nu) * (1 - 2 * nu))

    energy_positive = 0.5 * lame * (0.5 * (strain.trace() + abs(strain.trace()))) ** 2 + \
                      mu * (strain_positive * strain_positive).trace()

    energy_negative = 0.5 * lame * (0.5 * (strain.trace() - abs(strain.trace()))) ** 2 + \
                      mu * (strain_negative * strain_negative).trace()

    return energy_positive, energy_negative
