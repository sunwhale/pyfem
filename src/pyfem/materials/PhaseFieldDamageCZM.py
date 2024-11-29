# -*- coding: utf-8 -*-
"""

"""
from pyfem.io.Material import Material
from pyfem.io.Section import Section
from pyfem.materials.BaseMaterial import BaseMaterial
from pyfem.utils.colors import error_style


class PhaseFieldDamageCZM(BaseMaterial):
    """
    相场断裂材料。

    支持的截面属性：('Volume', 'PlaneStress', 'PlaneStrain')

    :ivar gc: surface energy to create a unit fracture surface gc
    :vartype gc: float

    :ivar lc: length scale parameter to measure the damage diffusion lc
    :vartype lc: float
    """

    __slots_dict__: dict = {
        'gc': ('float', 'surface energy to create a unit fracture surface gc'),
        'lc': ('float', 'length scale parameter to measure the damage diffusion lc'),
        'a1': ('float', 'degradation function coefficient a1'),
        'a2': ('float', 'degradation function coefficient a2'),
        'a3': ('float', 'degradation function coefficient a3'),
        'p': ('float', 'polynomial power p'),
        'xi': ('float', 'geometric function coefficient xi'),
        'c0': ('float', 'band width c0'),
    }

    __slots__ = BaseMaterial.__slots__ + [slot for slot in __slots_dict__.keys()]

    __data_keys__ = ['surface energy to create a unit fracture surface gc',
                     'length scale parameter to measure the damage diffusion lc',
                     'degradation function coefficient a1',
                     'degradation function coefficient a2',
                     'degradation function coefficient a3',
                     'polynomial power p',
                     'geometric function coefficient xi',
                     'band width c0']

    def __init__(self, material: Material, dimension: int, section: Section) -> None:
        super().__init__(material, dimension, section)
        self.allowed_section_types = ('Volume', 'PlaneStress', 'PlaneStrain')

        self.data_keys = self.__data_keys__

        if len(self.material.data) != len(self.data_keys):
            raise NotImplementedError(error_style(self.get_data_length_error_msg()))
        else:
            for i, key in enumerate(self.data_keys):
                self.data_dict[key] = material.data[i]

        self.gc: float = self.data_dict['surface energy to create a unit fracture surface gc']
        self.lc: float = self.data_dict['length scale parameter to measure the damage diffusion lc']
        self.a1: float = self.data_dict['degradation function coefficient a1']
        self.a2: float = self.data_dict['degradation function coefficient a2']
        self.a3: float = self.data_dict['degradation function coefficient a3']
        self.p: float = self.data_dict['polynomial power p']
        self.xi: float = self.data_dict['geometric function coefficient xi']
        self.c0: float = self.data_dict['band width c0']

        self.create_tangent()


if __name__ == "__main__":
    from pyfem.utils.visualization import print_slots_dict

    print_slots_dict(PhaseFieldDamageCZM.__slots_dict__)
