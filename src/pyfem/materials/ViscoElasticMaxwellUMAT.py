# -*- coding: utf-8 -*-
"""

"""
from copy import deepcopy

from numpy import zeros, ndarray, exp

from pyfem.fem.Timer import Timer
from pyfem.fem.constants import DTYPE
from pyfem.io.Material import Material
from pyfem.io.Section import Section
from pyfem.materials.BaseMaterial import BaseMaterial
from pyfem.materials.ElasticIsotropic import get_stiffness_from_young_poisson
from pyfem.utils.colors import error_style


class ViscoElasticMaxwell(BaseMaterial):
    """
    广义Maxwell粘弹性材料。

    :ivar E0: 弹性单元的弹性模量
    :vartype E0: float

    :ivar E1: 第1个粘弹性单元的弹性模量
    :vartype E1: float

    :ivar E2: 第2个粘弹性单元的弹性模量
    :vartype E2: float

    :ivar E3: 第3个粘弹性单元的弹性模量
    :vartype E3: float

    :ivar TAU1: 第1个粘弹性单元的时间系数
    :vartype TAU1: float

    :ivar TAU2: 第2个粘弹性单元的时间系数
    :vartype TAU2: float

    :ivar TAU3: 第3个粘弹性单元的时间系数
    :vartype TAU3: float

    :ivar POISSON: 泊松比
    :vartype POISSON: float
    """

    __slots_dict__: dict = {
        'E0': ('float', '弹性单元的弹性模量'),
        'E1': ('float', '第1个粘弹性单元的弹性模量'),
        'E2': ('float', '第2个粘弹性单元的弹性模量'),
        'E3': ('float', '第3个粘弹性单元的弹性模量'),
        'TAU1': ('float', '第1个粘弹性单元的时间系数'),
        'TAU2': ('float', '第2个粘弹性单元的时间系数'),
        'TAU3': ('float', '第3个粘弹性单元的时间系数'),
        'POISSON': ('float', '泊松比')
    }

    __slots__ = BaseMaterial.__slots__ + [slot for slot in __slots_dict__.keys()]

    def __init__(self, material: Material, dimension: int, section: Section) -> None:
        super().__init__(material, dimension, section)
        self.allowed_section_types = ('Volume', 'PlaneStress', 'PlaneStrain')

        self.data_keys = ['E0', 'Poisson\'s ratio nu', 'E1', 'TAU1', 'E2', 'TAU2', 'E3', 'TAU3']

        if len(self.material.data) != len(self.data_keys):
            raise NotImplementedError(error_style(self.get_data_length_error_msg()))
        else:
            for i, key in enumerate(self.data_keys):
                self.data_dict[key] = material.data[i]

        self.E0: float = self.data_dict['E0']
        self.E1: float = self.data_dict['E1']
        self.E2: float = self.data_dict['E2']
        self.E3: float = self.data_dict['E3']
        self.TAU1: float = self.data_dict['TAU1']
        self.TAU2: float = self.data_dict['TAU2']
        self.TAU3: float = self.data_dict['TAU3']
        self.POISSON: float = self.data_dict['Poisson\'s ratio nu']

        self.create_tangent()

    def create_tangent(self):
        if self.section.type in self.allowed_section_types:
            self.tangent = get_stiffness_from_young_poisson(self.dimension, self.E0, self.POISSON, self.section.type)
        else:
            raise NotImplementedError(error_style(self.get_section_type_error_msg()))

    def get_tangent(self, variable: dict[str, ndarray],
                    state_variable: dict[str, ndarray],
                    state_variable_new: dict[str, ndarray],
                    element_id: int,
                    igp: int,
                    ntens: int,
                    ndi: int,
                    nshr: int,
                    timer: Timer) -> tuple[ndarray, dict[str, ndarray]]:

        if state_variable == {}:
            state_variable['SM1'] = zeros(ntens, dtype=DTYPE)
            state_variable['SM2'] = zeros(ntens, dtype=DTYPE)
            state_variable['SM3'] = zeros(ntens, dtype=DTYPE)

        SM1OLD = deepcopy(state_variable['SM1'])
        SM2OLD = deepcopy(state_variable['SM2'])
        SM3OLD = deepcopy(state_variable['SM3'])

        dtime = timer.dtime
        dstrain = variable['dstrain']
        strain = variable['strain']

        E0 = self.E0
        E1 = self.E1
        E2 = self.E2
        E3 = self.E3
        TAU1 = self.TAU1
        TAU2 = self.TAU2
        TAU3 = self.TAU3
        POISSON = self.POISSON

        mu0 = 0.5 * E0 / (1.0 + POISSON)
        mu1 = 0.5 * E1 / (1.0 + POISSON)
        mu2 = 0.5 * E2 / (1.0 + POISSON)
        mu3 = 0.5 * E3 / (1.0 + POISSON)

        BULK = E0 / 3 / (1.0 - 2 * POISSON)

        m1 = (TAU1 * mu1 - TAU1 * mu1 * exp(-dtime / TAU1)) / (mu0 * dtime)
        m2 = (TAU2 * mu2 - TAU2 * mu2 * exp(-dtime / TAU2)) / (mu0 * dtime)
        m3 = (TAU3 * mu3 - TAU3 * mu3 * exp(-dtime / TAU3)) / (mu0 * dtime)

        term1 = BULK + (4.0 * mu0) / 3.0
        term2 = BULK - (2.0 * mu0) / 3.0

        g = zeros((ntens, ntens), dtype=DTYPE)

        for i in range(ndi):
            g[i, i] = term1
        for i in range(1, ndi):
            for j in range(0, i):
                g[i, j] = term2
                g[j, i] = term2
        for i in range(ndi, ntens):
            g[i, i] = mu0

        stress = zeros(ntens, dtype=DTYPE)

        for i in range(ndi):
            stress[i] = g[i, 0] * strain[0] + g[i, 1] * strain[1] + g[i, 2] * strain[2] + \
                        SM1OLD[i] + SM2OLD[i] + SM3OLD[i] + (1 + m1 + m2 + m3) * \
                        (g[i, 0] * dstrain[0] + g[i, 1] * dstrain[1] + g[i, 2] * dstrain[2])

        for i in range(ndi, ntens):
            stress[i] = g[i, i] * strain[i] + SM1OLD[i] + SM2OLD[i] + SM3OLD[i] + (1 + m1 + m2 + m3) * (
                    g[i, i] * dstrain[i])

        if element_id == 0 and igp == 0:
            print(stress)

        SM1 = zeros(ntens, dtype=DTYPE)
        SM2 = zeros(ntens, dtype=DTYPE)
        SM3 = zeros(ntens, dtype=DTYPE)

        for i in range(ndi):
            SM1[i] = SM1OLD[i] + m1 * g[i, 0] * dstrain[0] + g[i, 1] * dstrain[1] + g[i, 2] * dstrain[2]
            SM2[i] = SM2OLD[i] + m2 * g[i, 0] * dstrain[0] + g[i, 1] * dstrain[1] + g[i, 2] * dstrain[2]
            SM3[i] = SM3OLD[i] + m3 * g[i, 0] * dstrain[0] + g[i, 1] * dstrain[1] + g[i, 2] * dstrain[2]

        for i in range(ndi, ntens):
            SM1[i] = SM1OLD[i] + m1 * (g[i, i] * dstrain[i])
            SM2[i] = SM2OLD[i] + m2 * (g[i, i] * dstrain[i])
            SM3[i] = SM3OLD[i] + m3 * (g[i, i] * dstrain[i])

        for i in range(ntens):
            SM1[i] = exp(-dtime / TAU1) * SM1[i]
            SM2[i] = exp(-dtime / TAU2) * SM2[i]
            SM3[i] = exp(-dtime / TAU3) * SM3[i]

        state_variable_new['SM1'] = SM1
        state_variable_new['SM2'] = SM2
        state_variable_new['SM3'] = SM3

        ddsdde = (1 + m1 + m2 + m3) * g

        output = {'stress': stress}

        return ddsdde, output


if __name__ == "__main__":
    from pyfem.io.Properties import Properties

    props = Properties()
    props.read_file(r'..\..\..\examples\mechanical\1element\hex8\Job-1.toml')
    material_data = ViscoElasticMaxwell(props.materials[2], 3, props.sections[0])
    material_data.show()
