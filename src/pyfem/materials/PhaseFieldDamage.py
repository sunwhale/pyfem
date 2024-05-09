# -*- coding: utf-8 -*-
"""

"""
from pyfem.io.Material import Material
from pyfem.io.Section import Section
from pyfem.materials.BaseMaterial import BaseMaterial
from pyfem.utils.colors import error_style


class PhaseFieldDamage(BaseMaterial):
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
    }

    __slots__ = BaseMaterial.__slots__ + [slot for slot in __slots_dict__.keys()]

    __data_keys__ = ['surface energy to create a unit fracture surface gc',
                     'length scale parameter to measure the damage diffusion lc']

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

        self.create_tangent()


if __name__ == "__main__":
    from pyfem.utils.visualization import print_slots_dict

    print_slots_dict(PhaseFieldDamage.__slots_dict__)

    from pyfem.io.Properties import Properties

    props = Properties()
    props.read_file(r'..\..\..\examples\mechanical_phase\rectangle\Job-1.toml')
    material_data = PhaseFieldDamage(props.materials[1], 3, props.sections[0])
    material_data.show()
