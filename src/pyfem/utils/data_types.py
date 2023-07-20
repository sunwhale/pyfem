# -*- coding: utf-8 -*-
"""
定义数据类型注释变量。
"""

from typing import Union

from pyfem.materials.BaseMaterial import BaseMaterial
from pyfem.materials.ElasticIsotropic import ElasticIsotropic
from pyfem.materials.MechanicalThermalExpansion import MechanicalThermalExpansion
from pyfem.materials.PhaseFieldDamage import PhaseFieldDamage
from pyfem.materials.PlasticKinematicHardening import PlasticKinematicHardening
from pyfem.materials.ThermalIsotropic import ThermalIsotropic
from pyfem.materials.ViscoElasticMaxwell import ViscoElasticMaxwell

MaterialData = Union[
    BaseMaterial, MechanicalThermalExpansion, PhaseFieldDamage, PlasticKinematicHardening, ThermalIsotropic, ViscoElasticMaxwell, ElasticIsotropic]
