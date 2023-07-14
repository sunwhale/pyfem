# -*- coding: utf-8 -*-
"""
定义色彩关键字，用于对字符串着色。
"""

from typing import Union

from pyfem.materials.BaseMaterial import BaseMaterial
from pyfem.materials.MechanicalThermalExpansion import MechanicalThermalExpansion
from pyfem.materials.PhaseFieldDamage import PhaseFieldDamage
from pyfem.materials.PlasticKinematicHardening import PlasticKinematicHardening
from pyfem.materials.ThermalIsotropic import ThermalIsotropic
from pyfem.materials.ViscoElasticMaxwell import ViscoElasticMaxwell
from pyfem.materials.ElasticIsotropic import ElasticIsotropic

MaterialData = Union[BaseMaterial, MechanicalThermalExpansion, PhaseFieldDamage, PlasticKinematicHardening, ThermalIsotropic, ViscoElasticMaxwell, ElasticIsotropic]
