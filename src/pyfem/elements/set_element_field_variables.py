# -*- coding: utf-8 -*-
"""

"""
from numpy import ndarray, average, dot, tile


def set_element_field_variables(qp_field_variables: dict[str, ndarray], iso_element_shape, dimension: int) -> dict[str, ndarray]:
    """

    """
    method = ''

    element_nodal_values = {}
    for key, item in qp_field_variables.items():
        if iso_element_shape.extrapolated_matrix.shape != (0,) and method == 'extrapolated':
            if len(item.shape) == 1:
                element_nodal_values[key] = dot(iso_element_shape.extrapolated_matrix, item.reshape(-1, 1))
            else:
                element_nodal_values[key] = dot(iso_element_shape.extrapolated_matrix, item)
        else:
            element_nodal_values[key] = tile(average(item, axis=0), (iso_element_shape.nodes_number, 1))

    element_nodal_field_variables = {}
    for key, item in element_nodal_values.items():
        if key == 'stress':
            if dimension == 2:
                element_nodal_field_variables['S11'] = item[:, 0]
                element_nodal_field_variables['S22'] = item[:, 1]
                element_nodal_field_variables['S12'] = item[:, 2]

            elif dimension == 3:
                element_nodal_field_variables['S11'] = item[:, 0]
                element_nodal_field_variables['S22'] = item[:, 1]
                element_nodal_field_variables['S33'] = item[:, 2]
                element_nodal_field_variables['S12'] = item[:, 3]
                element_nodal_field_variables['S13'] = item[:, 4]
                element_nodal_field_variables['S23'] = item[:, 5]

            element_nodal_field_variables['S'] = item

        if key == 'strain':
            if dimension == 2:
                element_nodal_field_variables['E11'] = item[:, 0]
                element_nodal_field_variables['E22'] = item[:, 1]
                element_nodal_field_variables['E12'] = item[:, 2]

            elif dimension == 3:
                element_nodal_field_variables['E11'] = item[:, 0]
                element_nodal_field_variables['E22'] = item[:, 1]
                element_nodal_field_variables['E33'] = item[:, 2]
                element_nodal_field_variables['E12'] = item[:, 3]
                element_nodal_field_variables['E13'] = item[:, 4]
                element_nodal_field_variables['E23'] = item[:, 5]

            element_nodal_field_variables['E'] = item

        if key == 'energy':
            element_nodal_field_variables['Energy'] = item[:, 0]

        if key == 'heat_flux':
            if dimension >= 1:
                element_nodal_field_variables['HFL1'] = item[:, 0]
            if dimension >= 2:
                element_nodal_field_variables['HFL2'] = item[:, 1]
            if dimension >= 3:
                element_nodal_field_variables['HFL3'] = item[:, 2]

        if key == 'concentration_flux':
            if dimension >= 1:
                element_nodal_field_variables['CFL1'] = item[:, 0]
            if dimension >= 2:
                element_nodal_field_variables['CFL2'] = item[:, 1]
            if dimension >= 3:
                element_nodal_field_variables['CFL3'] = item[:, 2]

    return element_nodal_field_variables


if __name__ == "__main__":
    pass
