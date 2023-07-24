# -*- coding: utf-8 -*-
"""

"""
from copy import deepcopy

from numpy import zeros, ndarray, exp, dot, insert, delete

from pyfem.fem.Timer import Timer
from pyfem.fem.constants import DTYPE
from pyfem.io.Material import Material
from pyfem.io.Section import Section
from pyfem.materials.BaseMaterial import BaseMaterial
from pyfem.materials.ElasticIsotropic import get_stiffness_from_young_poisson
from pyfem.utils.colors import error_style


class ViscoElasticMaxwell(BaseMaterial):
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

    本构方程的一维标量形式：

    .. math::
        \sigma \left( t \right) = {E_0 }{\varepsilon _0} + \sum\limits_{i = 1}^n {\left( {{E_i}{e^{ - \frac{t}{{{\tau _i}}}}}} \right){\varepsilon _0} + } {E_0 }\left( {\varepsilon (t) - {\varepsilon _0}} \right) + \int_0^t {\sum\limits_{i = 1}^n {\left( {{E_i}{e^{ - \frac{{t - \theta }}{{{\tau _i}}}}}} \right)} \frac{{{\text{d}}\varepsilon (\theta )}}{{{\text{d}}s}}{\text{d}}\theta }

    本构方程的三维离散格式：

    .. math::
        \left\{ {\sigma \left( {{t_{n + 1}}} \right)} \right\} = \left[ {{C^{\text{e}}}} \right]\{ \varepsilon \left( {{t_n}} \right)\}  + \sum\limits_{i = 1}^N {{e^{ - \frac{{\Delta {t_{n + 1}}}}{{{\tau _i}}}}}{h_i}\left( {{t_n}} \right)}  + \left[ {1 + \sum\limits_{i = 1}^N {{\gamma _i}{\tau _i}\left( {1 - {e^{ - \frac{{\Delta {t_{n + 1}}}}{{{\tau _i}}}}}} \right)} } \right]\left[ {{C^{\text{e}}}} \right]\left\{ {\Delta \varepsilon \left( {{t_{n + 1}}} \right)} \right\}

    其中，

    .. math::
        \left\{ \begin{gathered}
          {\tau _i} = \frac{{{\eta _i}}}{{{E_i}}} \hfill \\
          {\gamma _i} = \frac{{{E_i}}}{{{E_0}}} \hfill \\
          {h_i}\left( t \right) = {E_i}\int_0^t {{e^{ - \frac{{t - s}}{{{\tau _i}}}}}\frac{{{\text{d}}\varepsilon }}{{{\text{d}}s}}{\text{d}}s}  \hfill \\
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

    """

    __slots_dict__: dict = {
        'E0': ('float', '弹性单元的弹性模量'),
        'E1': ('float', '第1个粘弹性单元的弹性模量'),
        'E2': ('float', '第2个粘弹性单元的弹性模量'),
        'E3': ('float', '第3个粘弹性单元的弹性模量'),
        'TAU1': ('float', '第1个粘弹性单元的时间系数'),
        'TAU2': ('float', '第2个粘弹性单元的时间系数'),
        'TAU3': ('float', '第3个粘弹性单元的时间系数'),
        'nu': ('float', '泊松比')
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
        self.nu: float = self.data_dict['Poisson\'s ratio nu']

        self.create_tangent()

    def create_tangent(self):
        if self.section.type in self.allowed_section_types:
            self.tangent = get_stiffness_from_young_poisson(self.dimension, self.E0, self.nu, self.section.type)
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
            state_variable['h1'] = zeros(ntens, dtype=DTYPE)
            state_variable['h2'] = zeros(ntens, dtype=DTYPE)
            state_variable['h3'] = zeros(ntens, dtype=DTYPE)

        h1 = deepcopy(state_variable['h1'])
        h2 = deepcopy(state_variable['h2'])
        h3 = deepcopy(state_variable['h3'])

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
            strain = insert(strain, 2, 0)
            dstrain = insert(dstrain, 2, 0)
        elif self.section.type == 'PlaneStress':
            strain = insert(strain, 2, -nu / (1 - nu) * (strain[0] + strain[1]))
            dstrain = insert(dstrain, 2, -nu / (1 - nu) * (dstrain[0] + dstrain[1]))

        mu0 = 0.5 * E0 / (1.0 + nu)
        bulk = E0 / 3.0 / (1.0 - 2 * nu)
        
        term1 = bulk + (4.0 * mu0) / 3.0
        term2 = bulk - (2.0 * mu0) / 3.0

        C = zeros((ntens, ntens), dtype=DTYPE)

        for i in range(ndi):
            C[i, i] = term1
        for i in range(1, ndi):
            for j in range(0, i):
                C[i, j] = term2
                C[j, i] = term2
        for i in range(ndi, ntens):
            C[i, i] = mu0

        a1 = exp(-dtime / TAU1)
        a2 = exp(-dtime / TAU2)
        a3 = exp(-dtime / TAU3)

        m1 = TAU1 * E1 / E0 * (1.0 - a1) / dtime
        m2 = TAU2 * E2 / E0 * (1.0 - a2) / dtime
        m3 = TAU3 * E3 / E0 * (1.0 - a3) / dtime

        term3 = 1 + m1 + m2 + m3
        term4 = dot(C, dstrain)

        stress = dot(C, strain) + h1 * a1 + h2 * a2 + h3 * a3 + term3 * term4

        h1 = h1 * a1 + m1 * term4
        h2 = h2 * a2 + m2 * term4
        h3 = h3 * a3 + m3 * term4

        state_variable_new['h1'] = h1
        state_variable_new['h2'] = h2
        state_variable_new['h3'] = h3

        ddsdde = (1 + m1 + m2 + m3) * C

        if self.section.type == 'PlaneStrain':
            ddsdde = delete(delete(ddsdde, 2, axis=0), 2, axis=1)
            stress = delete(stress, 2)
        elif self.section.type == 'PlaneStress':
            lam = E0 * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
            mu = E0 / (2.0 * (1.0 + nu))
            ddsdde = delete(delete(ddsdde, 2, axis=0), 2, axis=1)
            ddsdde[0, 0] -= lam * lam / (lam + 2 * mu)
            ddsdde[0, 1] -= lam * lam / (lam + 2 * mu)
            ddsdde[1, 0] -= lam * lam / (lam + 2 * mu)
            ddsdde[1, 1] -= lam * lam / (lam + 2 * mu)
            stress = delete(stress, 2)

        # if element_id == 0 and igp == 0:
        #     print(stress)

        output = {'stress': stress}

        return ddsdde, output


if __name__ == "__main__":
    from pyfem.utils.visualization import print_slots_dict

    print_slots_dict(ViscoElasticMaxwell.__slots_dict__)

    from pyfem.Job import Job

    job = Job(r'..\..\..\examples\mechanical\specimen3D\Job-1.toml')

    job.run()
