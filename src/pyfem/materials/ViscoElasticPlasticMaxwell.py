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


class ViscoElasticPlasticMaxwell(BaseMaterial):
    r"""
    广义Maxwell粘弹性材料。

    支持的截面属性：('Volume', 'PlaneStress', 'PlaneStrain')

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

    :ivar nu: 泊松比
    :vartype nu: float

    本构方程的一维标量形式[1]：

    .. math::
        \sigma \left( t \right) = {E_0}{\varepsilon _0} + \sum\limits_{i = 1}^n {\left( {{E_i}{e^{ - \frac{t}{{{\tau _i}}}}}} \right){\varepsilon _0} + } {E_0}\left[ {\varepsilon \left( t \right) - {\varepsilon _0}} \right] + \int_0^t {\sum\limits_{i = 1}^n {\left( {{E_i}{e^{ - \frac{{t - s}}{{{\tau _i}}}}}} \right)} \frac{{\partial \varepsilon \left( s \right)}}{{\partial s}}{\text{d}}s}

    本构方程的三维离散格式：

    .. math::
        \left\{ {\sigma \left( {{t_{n + 1}}} \right)} \right\} = \left[ {{C^{\text{e}}}} \right]\{ \varepsilon \left( {{t_n}} \right)\}  + \sum\limits_{i = 1}^N {{e^{ - \frac{{\Delta {t_{n + 1}}}}{{{\tau _i}}}}}{h_i}\left( {{t_n}} \right)}  + \left[ {1 + \sum\limits_{i = 1}^N {{\gamma _i}{\tau _i}\left( {1 - {e^{ - \frac{{\Delta {t_{n + 1}}}}{{{\tau _i}}}}}} \right)} } \right]\left[ {{C^{\text{e}}}} \right]\left\{ {\Delta \varepsilon \left( {{t_{n + 1}}} \right)} \right\}

    其中，

    .. math::
        \left\{ \begin{gathered}
          {\tau _i} = \frac{{{\eta _i}}}{{{E_i}}} \hfill \\
          {\gamma _i} = \frac{{{E_i}}}{{{E_0}}} \hfill \\
          {h_i}\left( t \right) = {E_i}\int_0^t {{e^{ - \frac{{t - s}}{{{\tau _i}}}}}\frac{{\partial \varepsilon \left( s \right)}}{{\partial s}}{\text{d}}s}  \hfill \\
        \end{gathered}  \right.

    .. math::
        \left[ {{C^{\text{e}}}} \right] = \left[ {\begin{array}{*{20}{c}}
          {K + \frac{4}{3}{\mu _0}}&{K - \frac{2}{3}{\mu _0}}&{K - \frac{2}{3}{\mu _0}}&0&0&0 \\
          {K - \frac{2}{3}{\mu _0}}&{K + \frac{4}{3}{\mu _0}}&{K - \frac{2}{3}{\mu _0}}&0&0&0 \\
          {K - \frac{2}{3}{\mu _0}}&{K - \frac{2}{3}{\mu _0}}&{K + \frac{4}{3}{\mu _0}}&0&0&0 \\
          0&0&0&{{\mu _0}}&0&0 \\
          0&0&0&0&{{\mu _0}}&0 \\
          0&0&0&0&0&{{\mu _0}}
        \end{array}} \right]

    编写get_tangent函数时用到的积分格式：

    .. math::
        h_i^{n + 1} = {e^{ - \frac{{\Delta t}}{{{\tau _i}}}}}h_i^n + {\gamma _i}\frac{{\left( {1 - {e^{ - \frac{{\Delta t}}{{{\tau _i}}}}}} \right)}}{{\frac{{\Delta t}}{{{\tau _i}}}}}{{\mathbf{C}}^{\text{e}}}{\text{:}}\Delta {{\mathbf{\varepsilon }}^{n + 1}}

    .. math::
        {{\mathbf{\sigma }}^{n + 1}} = {{\mathbf{C}}^{\text{e}}}{\text{:}}{{\mathbf{\varepsilon }}^n} + \sum\limits_{i = 1}^N {{e^{ - \frac{{\Delta t}}{{{\tau _i}}}}}} h_i^n + \left[ {1 + \sum\limits_{i = 1}^N {\frac{{{\gamma _i}{\tau _i}}}{{\Delta t}}\left( {1 - {e^{ - \frac{{\Delta t}}{{{\tau _i}}}}}} \right)} } \right]{{\mathbf{C}}^{\text{e}}}{\text{:}}\Delta {{\mathbf{\varepsilon }}^{n + 1}}

    材料的一致性刚度矩阵 ddsdde：

    .. math::
        \frac{{\partial \Delta {\mathbf{\sigma }}}}{{\partial \Delta {\mathbf{\varepsilon }}}} = \left[ {1 + \sum\limits_{i = 1}^N {\frac{{{\gamma _i}{\tau _i}}}{{\Delta t}}\left( {1 - {e^{ - \frac{{\Delta t}}{{{\tau _i}}}}}} \right)} } \right]{{\mathbf{C}}^{\text{e}}}

    [1] Gillani A. Development of Material Model Subroutines for Linear and Nonlinear Response of Elastomers[D]. The University of Western Ontario (Canada), 2018.

    """

    __slots_dict__: dict = {
        'E0': ('float', '弹性单元的弹性模量'),
        'E1': ('float', '第1个粘弹性单元的弹性模量'),
        'E2': ('float', '第2个粘弹性单元的弹性模量'),
        'E3': ('float', '第3个粘弹性单元的弹性模量'),
        'TAU1': ('float', '第1个粘弹性单元的时间系数'),
        'TAU2': ('float', '第2个粘弹性单元的时间系数'),
        'TAU3': ('float', '第3个粘弹性单元的时间系数'),
        'nu': ('float', '泊松比'),
        'yield_stress': ('float', 'Yield stress'),
        'hard': ('float', 'Hardening coefficient'),
        'EBULK3': ('float', '3倍体积模量'),
        'EG': ('float', '剪切模量'),
        'EG2': ('float', '2倍剪切模量'),
        'EG3': ('float', '3倍剪切模量'),
        'ELAM': ('float', '拉梅常数'),
        'tolerance': ('float', '判断屈服的误差容限'),
    }

    __slots__ = BaseMaterial.__slots__ + [slot for slot in __slots_dict__.keys()]

    __data_keys__ = ['E0', 'Poisson\'s ratio nu', 'E1', 'TAU1', 'E2', 'TAU2', 'E3', 'TAU3']

    def __init__(self, material: Material, dimension: int, section: Section) -> None:
        super().__init__(material, dimension, section)
        self.allowed_section_types = ('Volume', 'PlaneStress', 'PlaneStrain')

        self.data_keys = self.__data_keys__

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
        self.nu: float = self.data_dict['Poisson\'s ratio nu']

        self.yield_stress: float = 1.0
        self.hard: float = 0.2

        E = self.E0
        self.EBULK3: float = E / (1.0 - 2.0 * self.nu)
        self.EG2: float = E / (1.0 + self.nu)
        self.EG: float = self.EG2 / 2.0
        self.EG3: float = 3.0 * self.EG
        self.ELAM: float = (self.EBULK3 - self.EG2) / 3.0
        self.tolerance: float = 1.0e-10

        self.create_tangent()

    def create_tangent(self):
        if self.section.type in self.allowed_section_types:
            self.tangent = get_stiffness_from_young_poisson(self.dimension, self.E0, self.nu, self.section.type)
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
            state_variable['h1'] = np.zeros(ntens, dtype=DTYPE)
            state_variable['h2'] = np.zeros(ntens, dtype=DTYPE)
            state_variable['h3'] = np.zeros(ntens, dtype=DTYPE)
            state_variable['elastic_strain'] = np.zeros(ntens, dtype=DTYPE)
            state_variable['plastic_strain'] = np.zeros(ntens, dtype=DTYPE)
            state_variable['back_stress'] = np.zeros(ntens, dtype=DTYPE)
            state_variable['stress'] = np.zeros(ntens, dtype=DTYPE)

        h1 = deepcopy(state_variable['h1'])
        h2 = deepcopy(state_variable['h2'])
        h3 = deepcopy(state_variable['h3'])
        elastic_strain = deepcopy(state_variable['elastic_strain'])
        plastic_strain = deepcopy(state_variable['plastic_strain'])
        back_stress = deepcopy(state_variable['back_stress'])
        stress = deepcopy(state_variable['stress'])

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
        nu = self.nu

        if self.section.type == 'PlaneStrain':
            strain = np.insert(strain, 2, 0)
            dstrain = np.insert(dstrain, 2, 0)
        elif self.section.type == 'PlaneStress':
            strain = np.insert(strain, 2, -nu / (1 - nu) * (strain[0] + strain[1]))
            dstrain = np.insert(dstrain, 2, -nu / (1 - nu) * (dstrain[0] + dstrain[1]))

        mu0 = 0.5 * E0 / (1.0 + nu)
        bulk = E0 / 3.0 / (1.0 - 2.0 * nu)

        term1 = bulk + (4.0 * mu0) / 3.0
        term2 = bulk - (2.0 * mu0) / 3.0

        Ce = np.zeros((ntens, ntens), dtype=DTYPE)

        for i in range(ndi):
            Ce[i, i] = term1
        for i in range(1, ndi):
            for j in range(0, i):
                Ce[i, j] = term2
                Ce[j, i] = term2
        for i in range(ndi, ntens):
            Ce[i, i] = mu0

        a1 = np.exp(-dtime / TAU1)
        a2 = np.exp(-dtime / TAU2)
        a3 = np.exp(-dtime / TAU3)

        m1 = TAU1 * E1 / E0 * (1.0 - a1) / dtime
        m2 = TAU2 * E2 / E0 * (1.0 - a2) / dtime
        m3 = TAU3 * E3 / E0 * (1.0 - a3) / dtime

        term3 = 1.0 + m1 + m2 + m3
        term4 = np.dot(Ce, dstrain)

        stress = np.dot(Ce, strain) + h1 * a1 + h2 * a2 + h3 * a3 + term3 * term4

        s = stress - back_stress
        smises = self.get_smises(s)
        ddsddp = np.zeros(shape=(ntens, ntens), dtype=DTYPE)

        elastic_strain += dstrain

        if smises > (1.0 + self.tolerance) * self.yield_stress:
            hydrostatic_stress = sum(stress[:ndi]) / 3.0
            flow = stress - back_stress
            flow[:ndi] = flow[:ndi] - hydrostatic_stress
            flow *= 1.0 / smises

            delta_p = (smises - self.yield_stress) / (self.EG3 + self.hard)
            back_stress += self.hard * flow * delta_p

            plastic_strain[:ndi] += 1.5 * flow[:ndi] * delta_p
            elastic_strain[:ndi] -= 1.5 * flow[:ndi] * delta_p

            plastic_strain[ndi:] += 3.0 * flow[ndi:] * delta_p
            elastic_strain[ndi:] -= 3.0 * flow[ndi:] * delta_p

            stress = back_stress + flow * self.yield_stress
            stress[:ndi] += hydrostatic_stress

            EFFG = self.EG * (self.yield_stress + self.hard * delta_p) / smises
            EFFG2 = 2.0 * EFFG
            EFFG3 = 3.0 * EFFG
            EFFLAM = (self.EBULK3 - EFFG2) / 3.0
            EFFHRD = self.EG3 * self.hard / (self.EG3 + self.hard) - EFFG3

            ddsddp += EFFHRD * np.outer(flow, flow)

        state_variable_new['elastic_strain'] = elastic_strain
        state_variable_new['plastic_strain'] = plastic_strain
        state_variable_new['back_stress'] = back_stress
        state_variable_new['stress'] = stress

        h1 = h1 * a1 + m1 * term4
        h2 = h2 * a2 + m2 * term4
        h3 = h3 * a3 + m3 * term4

        state_variable_new['h1'] = h1
        state_variable_new['h2'] = h2
        state_variable_new['h3'] = h3

        ddsdde = (1.0 + m1 + m2 + m3) * Ce
        ddsdde += ddsddp

        strain_energy = 0.5 * sum(strain * stress)

        if self.section.type == 'PlaneStrain':
            ddsdde = np.delete(np.delete(ddsdde, 2, axis=0), 2, axis=1)
            stress = np.delete(stress, 2)
        elif self.section.type == 'PlaneStress':
            lam = E0 * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
            mu = E0 / (2.0 * (1.0 + nu))
            ddsdde = np.delete(np.delete(ddsdde, 2, axis=0), 2, axis=1)
            ddsdde[0, 0] -= lam * lam / (lam + 2 * mu)
            ddsdde[0, 1] -= lam * lam / (lam + 2 * mu)
            ddsdde[1, 0] -= lam * lam / (lam + 2 * mu)
            ddsdde[1, 1] -= lam * lam / (lam + 2 * mu)
            stress = np.delete(stress, 2)

        output = {'stress': stress, 'strain_energy': strain_energy}

        return ddsdde, output

    def get_smises(self, s: np.ndarray) -> float:
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

    print_slots_dict(ViscoElasticPlasticMaxwell.__slots_dict__)

    from pyfem.job.Job import Job

    job = Job(r'..\..\..\examples\mechanical\specimen3D\Job-1.toml')

    job.run()
