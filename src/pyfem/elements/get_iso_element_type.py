from numpy import ndarray

from pyfem.utils.colors import error_style


def get_iso_element_type(node_coords: ndarray) -> str:
    element_node_number = node_coords.shape[0]
    dimension = node_coords.shape[1]

    if dimension == 1:
        if element_node_number == 2:
            return 'line2'
        elif element_node_number == 3:
            return 'line3'
        else:
            error_msg = f'no 1D element with {element_node_number} nodes available'
            raise NotImplementedError(error_style(error_msg))
    elif dimension == 2:
        if element_node_number == 3:
            return 'tria3'
        elif element_node_number == 4:
            return 'quad4'
        elif element_node_number == 6:
            return 'tria6'
        elif element_node_number == 8:
            return 'quad8'
        elif element_node_number == 9:
            return 'quad9'
        else:
            error_msg = f'no 2D element with {element_node_number} nodes available'
            raise NotImplementedError(error_style(error_msg))
    elif dimension == 3:
        if element_node_number == 4:
            return 'tetra4'
        elif element_node_number == 5:
            return 'pyramid5'
        elif element_node_number == 6:
            return 'prism6'
        elif element_node_number == 8:
            return 'hexa8'
        elif element_node_number == 18:
            return 'prism18'
        elif element_node_number == 20:
            return 'hex20'
        else:
            error_msg = f'no 3D element with {element_node_number} nodes available'
            raise NotImplementedError(error_style(error_msg))
    else:
        error_msg = 'Dimension must be 1,2 or 3'
        raise NotImplementedError(error_style(error_msg))
