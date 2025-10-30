# -*- coding: utf-8 -*-
"""

"""
from pyfem.io.Material import Material
from pyfem.io.Section import Section
from pyfem.materials.BaseMaterial import BaseMaterial
from pyfem.utils.colors import error_style


class GradientPhaseFieldDamage(BaseMaterial):
    """
    相场断裂材料。

    支持的截面属性：('Volume', 'PlaneStress', 'PlaneStrain')

    :ivar g1c: surface energy to create a unit model I fracture surface g1c
    :vartype g1c: float

    :ivar g2c: surface energy to create a unit model II fracture surface g2c
    :vartype g2c: float

    :ivar lc: length scale parameter to measure the damage diffusion lc
    :vartype lc: float
    """

    __slots_dict__: dict = {
        'g1c': ('float', 'surface energy to create a unit model I fracture surface g1c'),
        'g2c': ('float', 'surface energy to create a unit model II fracture surface g2c'),
        'lc': ('float', 'length scale parameter to measure the damage diffusion lc'),
    }

    __slots__ = BaseMaterial.__slots__ + [slot for slot in __slots_dict__.keys()]

    __data_keys__ = ['surface energy to create a unit model I fracture surface g1c',
                     'surface energy to create a unit model II fracture surface g2c',
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

        self.g1c: float = self.data_dict['surface energy to create a unit model I fracture surface g1c']
        self.g2c: float = self.data_dict['surface energy to create a unit model II fracture surface g2c']
        self.lc: float = self.data_dict['length scale parameter to measure the damage diffusion lc']

        self.create_tangent()


if __name__ == "__main__":
    from pyfem.utils.visualization import print_slots_dict

    print_slots_dict(GradientPhaseFieldDamage.__slots_dict__)

    from pyfem.io.Properties import Properties

    props = Properties()
    props.read_file(r'..\..\..\examples\mechanical_phase\two_side_crack_gradient\Job-1.toml')
    material_data = GradientPhaseFieldDamage(props.materials[1], 3, props.sections[0])
    material_data.show()
