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

        Gc, q, r = self.Gc, self.q, self.r
        delta_n_max, delta_t_max = self.delta_n_max, self.delta_t_max

        epsilon_n, epsilon_t = strain1[0], strain1[1]
        n_norm = epsilon_n / delta_n_max
        t_norm = epsilon_t / delta_t_max

        exp_n = np.exp(-n_norm)
        exp_t = np.exp(-t_norm ** 2)

        stress[0] = (Gc * exp_n * n_norm / delta_n_max * (1.0 - q + q * exp_t))
        stress[1] = (2.0 * Gc * q * exp_n * (1.0 + n_norm) * epsilon_t * exp_t / (delta_t_max ** 2))

        # 辅助变量
        alpha = (r - q) / (r - 1.0)
        beta = (1 - q) / (r - 1.0)
        eta = q + alpha * n_norm

        # 预计算常用组合
        factor1 = Gc * exp_n
        factor2 = factor1 * exp_t
        factor3 = factor2 / delta_n_max

        # 雅可比矩阵
        ddsdde = np.zeros((2, 2))

        # (0, 0) 分量
        term1 = ((1.0 - r + n_norm) * beta - eta * exp_t) / delta_n_max ** 2
        term2 = 2.0 * (beta - alpha * exp_t) / (delta_n_max ** 2)
        ddsdde[0, 0] = factor1 * (term1 - term2)

        # (0, 1) 和 (1, 0) 分量
        common_term = factor3 * epsilon_t / delta_t_max ** 2 * (alpha - eta)
        ddsdde[0, 1] = ddsdde[1, 0] = 2.0 * common_term

        # (1, 1) 分量
        ddsdde[1, 1] = 2.0 * factor2 * eta / delta_t_max ** 2 * (1 - 2.0 * t_norm ** 2)

        np.set_printoptions(precision=5, suppress=True)
        # print('exp_n', exp_n, exp_t)
        # print('strain1', strain1)
        # print('stress', stress)
        print('ddsdde\n', ddsdde)
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
