# -*- coding: utf-8 -*-
"""

"""
from pyfem.io.Material import Material
from pyfem.io.Section import Section
from pyfem.materials.BaseMaterial import BaseMaterial
from pyfem.utils.colors import error_style


class PhaseFieldDamage(BaseMaterial):
    __slots__ = BaseMaterial.__slots__ + ('gc', 'lc')

    def __init__(self, material: Material, dimension: int, section: Section) -> None:
        super().__init__(material, dimension, section)
        self.allowed_section_types = ('Volume', 'PlaneStress', 'PlaneStrain')

        self.data_keys = ['surface energy to create a unit fracture surface gc',
                          'length scale parameter to measure the damage diffusion lc']

        if len(self.material.data) != len(self.data_keys):
            raise NotImplementedError(error_style(self.get_data_length_error_msg()))
        else:
            for i, key in enumerate(self.data_keys):
                self.data_dict[key] = material.data[i]

        self.gc: float = self.data_dict['surface energy to create a unit fracture surface gc']
        self.lc: float = self.data_dict['length scale parameter to measure the damage diffusion lc']

        self.create_tangent()


if __name__ == "__main__":
    from pyfem.io.Properties import Properties

    props = Properties()
    props.read_file(r'..\..\..\examples\mechanical_phase\rectangle\Job-1.toml')
    material_data = PhaseFieldDamage(props.materials[1], 3, props.sections[0])
    material_data.show()
