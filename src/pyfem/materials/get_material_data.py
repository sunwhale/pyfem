# -*- coding: utf-8 -*-
"""

"""
from typing import Union

from pyfem.io.Material import Material
from pyfem.io.Section import Section
from pyfem.materials.BaseMaterial import BaseMaterial
from pyfem.materials.DiffusionIsotropic import DiffusionIsotropic
from pyfem.materials.ElasticIsotropic import ElasticIsotropic
from pyfem.materials.MechanicalThermalExpansion import MechanicalThermalExpansion
from pyfem.materials.PhaseFieldDamage import PhaseFieldDamage
from pyfem.materials.PhaseFieldDamageCZM import PhaseFieldDamageCZM
from pyfem.materials.GradientPhaseFieldDamage import GradientPhaseFieldDamage
from pyfem.materials.PlasticCrystal import PlasticCrystal
from pyfem.materials.PlasticCrystalGNDs import PlasticCrystalGNDs
from pyfem.materials.PlasticIsotropicHardening import PlasticIsotropicHardening
from pyfem.materials.PlasticKinematicHardening import PlasticKinematicHardening
from pyfem.materials.ThermalIsotropic import ThermalIsotropic
from pyfem.materials.ViscoElasticMaxwell import ViscoElasticMaxwell
from pyfem.utils.colors import error_style

MaterialData = Union[
    BaseMaterial, MechanicalThermalExpansion, PhaseFieldDamage, PhaseFieldDamageCZM, PlasticIsotropicHardening, PlasticKinematicHardening, ThermalIsotropic, ViscoElasticMaxwell, ElasticIsotropic, DiffusionIsotropic]

material_data_dict = {
    'ElasticIsotropic': ElasticIsotropic,
    'PlasticIsotropicHardening': PlasticIsotropicHardening,
    'PlasticKinematicHardening': PlasticKinematicHardening,
    'PlasticCrystal': PlasticCrystal,
    'PlasticCrystalGNDs': PlasticCrystalGNDs,
    'ViscoElasticMaxwell': ViscoElasticMaxwell,
    'ThermalIsotropic': ThermalIsotropic,
    'MechanicalThermalExpansion': MechanicalThermalExpansion,
    'PhaseFieldDamage': PhaseFieldDamage,
    'PhaseFieldDamageCZM': PhaseFieldDamageCZM,
    'GradientPhaseFieldDamage': GradientPhaseFieldDamage,
    'DiffusionIsotropic': DiffusionIsotropic,
}


def get_material_data(material: Material, dimension: int, section: Section) -> MaterialData:
    """
    工厂函数，用于根据材料属性生产不同的材料对象。

    Args:
        material(Material): 材料属性
        dimension(int): 空间维度
        section(Section): 截面属性

    :return: 材料对象
    :rtype: MaterialData
    """

    if material.user_path is not None:
        import importlib.util
        spec = importlib.util.spec_from_file_location('User', material.user_path)
        user_material = importlib.util.module_from_spec(spec)  # type: ignore
        spec.loader.exec_module(user_material)  # type: ignore
        return user_material.User(material, dimension, section)

    else:
        class_name = f'{material.category}{material.type}'.strip().replace(' ', '')

        if class_name in material_data_dict:
            return material_data_dict[class_name](material, dimension, section)
        else:
            error_msg = f'{class_name} material is not supported.\n'
            error_msg += f'The allowed material types are {list(material_data_dict.keys())}.'
            raise NotImplementedError(error_style(error_msg))


if __name__ == "__main__":
    from pyfem.io.Properties import Properties

    props = Properties()
    props.read_file(r'..\..\..\examples\mechanical\plane\Job-1.toml')

    material_data = get_material_data(props.materials[0], 3, props.sections[0])
    material_data.show()
