# -*- coding: utf-8 -*-
"""

"""
from copy import deepcopy

import numpy as np

from pyfem.fem.Timer import Timer
from pyfem.fem.constants import DTYPE
from pyfem.io.Material import Material
from pyfem.io.Section import Section
from pyfem.materials.BaseMaterial import BaseMaterial
from pyfem.materials.ElasticIsotropic import get_stiffness_from_young_poisson
from pyfem.utils.colors import error_style


class Cohesive(BaseMaterial):
    """
    随动强化塑性材料。

    支持的截面属性：('Volume', 'PlaneStress', 'PlaneStrain')

    :ivar E: Young's modulus E
    :vartype E: float

    :ivar nu: Poisson's ratio nu
    :vartype nu: float

    :ivar yield_stress: Yield stress
    :vartype yield_stress: float

    :ivar hard: Hardening coefficient
    :vartype hard: float

    :ivar EBULK3: 3倍体积模量
    :vartype EBULK3: float

    :ivar EG: 剪切模量
    :vartype EG: float

    :ivar EG2: 2倍剪切模量
    :vartype EG2: float

    :ivar EG3: 3倍剪切模量
    :vartype EG3: float

    :ivar ELAM: 拉梅常数
    :vartype ELAM: float

    :ivar tolerance: 判断屈服的误差容限
    :vartype tolerance: float
    """

    __slots_dict__: dict = {
        'E': ('float', 'Young\'s modulus E'),
        'nu': ('float', 'Poisson\'s ratio nu'),
        'T0': ('float', 'Yield stress'),
        'Gc': ('float', 'Hardening coefficient'),
        'r': ('float', '3倍体积模量'),
        'q': ('float', '剪切模量'),
        'delta_n_max': ('float', '2倍剪切模量'),
        'delta_t_max': ('float', '3倍剪切模量'),
    }

    __slots__ = BaseMaterial.__slots__ + [slot for slot in __slots_dict__.keys()]

    __data_keys__ = ['Young\'s modulus E', 'Poisson\'s ratio nu', 'Yield stress', 'Hardening coefficient']

    def __init__(self, material: Material, dimension: int, section: Section) -> None:
        super().__init__(material, dimension, section)
        self.allowed_section_types = ('CohesiveZone')

        self.data_keys = self.__data_keys__

        if len(self.material.data) != len(self.data_keys):
            raise NotImplementedError(error_style(self.get_data_length_error_msg()))
        else:
            for i, key in enumerate(self.data_keys):
                self.data_dict[key] = material.data[i]

        self.E: float = self.data_dict['Young\'s modulus E']
        self.nu: float = self.data_dict['Poisson\'s ratio nu']

        self.Gc: float = 0.1
        self.T0: float = 0.5

        self.r = 0.
        self.q = 1.
        self.delta_n_max = self.Gc / (2.71828183 * self.T0)
        self.delta_t_max = self.q * self.Gc / (1.16580058 * self.T0)

        self.create_tangent()

    def create_tangent(self):
        if self.section.type in self.allowed_section_types:
            self.tangent = get_stiffness_from_young_poisson(self.dimension, self.E, self.nu, self.section.type)
        else:
            raise NotImplementedError(error_style(self.get_section_type_error_msg()))

    def get_tangent(self, variable: dict[str, np.ndarray],
                    state_variable: dict[str, np.ndarray],
                    state_variable_new: dict[str, np.ndarray],
                    element_id: int,
                    iqp: int,
                    ntens: int,
                    ndi: int,
                    nshr: int,
                    timer: Timer) -> tuple[np.ndarray, dict[str, np.ndarray]]:

        if state_variable == {} or timer.time0 == 0.0:
            pass

        strain0 = variable['strain']
        dstrain = variable['dstrain']
        strain1 = strain0 + dstrain

        ddsdde = np.zeros((ntens, ntens), dtype=DTYPE)
        stress = np.zeros(ntens, dtype=DTYPE)

        # strain1 = [0., 0.]

        epsilon_n, epsilon_t = strain1[0], strain1[1]
        n_norm = epsilon_n / self.delta_n_max
        t_norm_sq = (epsilon_t / self.delta_t_max) ** 2

        exp_n = np.exp(-n_norm)
        exp_t = np.exp(-t_norm_sq)

        stress[0] = (self.Gc * exp_n * n_norm / self.delta_n_max * (1.0 - self.q + self.q * exp_t))
        stress[1] = (2.0 * self.Gc * self.q * exp_n * (1.0 + n_norm) * epsilon_t * exp_t / (self.delta_t_max * self.delta_t_max))

        t1 = self.delta_n_max * self.delta_n_max
        t4 = 1 / self.delta_n_max
        t5 = strain1[0] * t4
        t6 = np.exp(-t5)
        t8 = 1.0 - self.q
        t11 = 1 / (self.r - 1.0)
        t14 = (self.r - self.q) * t11
        t16 = self.q + t14 * t5
        t17 = strain1[1] * strain1[1]
        t18 = self.delta_t_max * self.delta_t_max
        t19 = 1 / t18
        t21 = np.exp(-t17 * t19)
        t26 = self.Gc * t4
        t38 = t19 * t21
        t41 = self.Gc * t6
        t46 = -t26 * t6 * t16 * strain1[1] * t38 + t41 * t14 * t4 * strain1[1] * t38
        t52 = t18 * t18

        ddsdde[0, 0] = self.Gc / t1 * t6 * ((1.0 - self.r + t5) * t8 * t11 - t16 * t21) - 2.0 * t26 * t6 * (t4 * t8 * t11 - t14 * t4 * t21)
        ddsdde[0, 1] = 2.0 * t46
        ddsdde[1, 0] = 2.0 * t46
        ddsdde[1, 1] = 2.0 * t41 * t16 * t19 * t21 - 4.0 * t41 * t16 * t17 / t52 * t21


        np.set_printoptions(precision=3, suppress=True)
        # print('exp_n', exp_n, exp_t)
        print('strain1', strain1)
        print('stress', stress)
        # print('ddsdde', ddsdde)
        output = {'stress': stress}

        return ddsdde, output


def get_smises(s: np.ndarray) -> float:
    if len(s) == 3:
        smises = np.sqrt(s[0] ** 2 + s[1] ** 2 - s[0] * s[1] + 3 * s[2] ** 2)
        return float(smises)
    elif len(s) >= 3:
        smises = (s[0] - s[1]) ** 2 + (s[1] - s[2]) ** 2 + (s[2] - s[0]) ** 2
        smises += 6 * sum([i ** 2 for i in s[3:]])
        smises = np.sqrt(0.5 * smises)
        return float(smises)
    else:
        raise NotImplementedError(error_style(f'unsupported stress dimension {len(s)}'))


if __name__ == "__main__":
    from pyfem.utils.visualization import print_slots_dict

    print_slots_dict(Cohesive.__slots_dict__)

    from pyfem.io.Properties import Properties

    props = Properties()
    props.read_file(r'..\..\..\examples\mechanical\plane\Job-1.toml')
    material_data = Cohesive(props.materials[0], 3, props.sections[0])
    material_data.show()
