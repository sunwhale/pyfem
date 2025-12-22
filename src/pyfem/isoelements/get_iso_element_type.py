# -*- coding: utf-8 -*-
"""

"""
import numpy as np

from pyfem.io.Section import Section
from pyfem.utils.colors import error_style


def get_iso_element_type(node_coords: np.ndarray, section: Section = None, dimension: int = -1) -> str:
    """
    根据单元节点坐标数组和单元空间维度返回默认的等参元名称。

    Args:
        node_coords(np.ndarray): 单元节点坐标数组
        dimension(int): 单元空间维度

    :return: 等参元名称
    :rtype: str
    """

    element_node_number = node_coords.shape[0]

    if dimension == -1:
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
            if section == None:
                return 'quad4'
            elif section.category == 'CohesiveZone':
                return 'line2_coh'
            else:
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
            return 'hex8'
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
