# -*- coding: utf-8 -*-
"""

"""
from typing import Union

import numpy as np

from pyfem.elements.BaseElement import BaseElement
from pyfem.elements.CohesiveZone import CohesiveZone
from pyfem.elements.Diffusion import Diffusion
from pyfem.elements.SolidFiniteStrain import SolidFiniteStrain
from pyfem.elements.SolidGradientPhaseDamageSmallStrain import SolidGradientPhaseDamageSmallStrain
from pyfem.elements.SolidPhaseDamageCZMSmallStrain import SolidPhaseDamageCZMSmallStrain
from pyfem.elements.SolidPhaseDamageDiffusionSmallStrain import SolidPhaseDamageDiffusionSmallStrain
from pyfem.elements.SolidPhaseDamageSmallStrain import SolidPhaseDamageSmallStrain
from pyfem.elements.SolidSmallStrain import SolidSmallStrain
from pyfem.elements.SolidThermalSmallStrain import SolidThermalSmallStrain
from pyfem.elements.Thermal import Thermal
from pyfem.fem.Timer import Timer
from pyfem.io.Dof import Dof
from pyfem.io.Material import Material
from pyfem.io.Section import Section
from pyfem.isoelements.IsoElementShape import IsoElementShape
from pyfem.materials.get_material_data import MaterialData
from pyfem.utils.colors import error_style

ElementData = Union[
    BaseElement, SolidSmallStrain, SolidThermalSmallStrain, SolidPhaseDamageSmallStrain, SolidGradientPhaseDamageSmallStrain, SolidPhaseDamageDiffusionSmallStrain, SolidPhaseDamageCZMSmallStrain, Thermal, Diffusion, CohesiveZone]

element_data_dict = {
    'SolidSmallStrain': SolidSmallStrain,
    'SolidFiniteStrain': SolidFiniteStrain,
    'SolidThermalSmallStrain': SolidThermalSmallStrain,
    'SolidPhaseDamageSmallStrain': SolidPhaseDamageSmallStrain,
    'SolidPhaseDamageCZMSmallStrain': SolidPhaseDamageCZMSmallStrain,
    'SolidPhaseDamageDiffusionSmallStrain': SolidPhaseDamageDiffusionSmallStrain,
    'SolidGradientPhaseDamageSmallStrain': SolidGradientPhaseDamageSmallStrain,
    'Thermal': Thermal,
    'Diffusion': Diffusion,
    'CohesiveZone': CohesiveZone,
}


def get_element_data(element_id: int,
                     iso_element_shape: IsoElementShape,
                     connectivity: np.ndarray,
                     node_coords: np.ndarray,
                     dof: Dof,
                     materials: list[Material],
                     section: Section,
                     material_data_list: list[MaterialData],
                     timer: Timer) -> ElementData:
    """
    工厂函数，用于根据材料、截面和单元属性生产不同的单元对象。

    Args:
        element_id(int): 单元编号
        iso_element_shape(IsoElementShape): 等参元对象
        connectivity(np.ndarray): 单元节点序列
        node_coords(np.ndarray): 单元坐标列表
        dof(Dof): 自由度属性
        materials(list[Material]): 材料属性列表
        section(Section): 截面属性
        material_data_list(list[MaterialData]): 材料数据对象列表
        timer(Timer): 计时器对象

    :return: 单元对象
    :rtype: ElementData
    """

    class_name = f'{section.category}{section.option}'.strip().replace(' ', '')

    if class_name in element_data_dict:
        return element_data_dict[class_name](element_id=element_id,
                                             iso_element_shape=iso_element_shape,
                                             connectivity=connectivity,
                                             node_coords=node_coords,
                                             dof=dof,
                                             materials=materials,
                                             section=section,
                                             material_data_list=material_data_list,
                                             timer=timer)
    else:
        error_msg = f'{class_name} element is not supported.\n'
        error_msg += f'The allowed element types are {list(element_data_dict.keys())}.'
        raise NotImplementedError(error_style(error_msg))


if __name__ == "__main__":
    pass
