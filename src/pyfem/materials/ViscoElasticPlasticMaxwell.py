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
            state_variable['epseq'] = np.zeros(1, dtype=DTYPE)
            state_variable['plastic_strain'] = np.zeros(ntens, dtype=DTYPE)
            state_variable['back_stress'] = np.zeros(ntens, dtype=DTYPE)
            state_variable['stress'] = np.zeros(ntens, dtype=DTYPE)

        h1 = deepcopy(state_variable['h1'])
        h2 = deepcopy(state_variable['h2'])
        h3 = deepcopy(state_variable['h3'])
        epseq = deepcopy(state_variable['epseq'])
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

        sigma_y = self.yield_stress
        H = 1.0
        beta = 0.0
        C = 1.0
        gamma = 1.0

        # 弹性刚度矩阵计算
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

        # 粘弹性参数
        a1 = np.exp(-dtime / TAU1)
        a2 = np.exp(-dtime / TAU2)
        a3 = np.exp(-dtime / TAU3)

        m1 = TAU1 * E1 / E0 * (1.0 - a1) / dtime
        m2 = TAU2 * E2 / E0 * (1.0 - a2) / dtime
        m3 = TAU3 * E3 / E0 * (1.0 - a3) / dtime

        term3 = 1.0 + m1 + m2 + m3

        # 计算粘弹性预测应力
        term4 = np.dot(Ce, dstrain)
        stress_pred = np.dot(Ce, strain) + h1 * a1 + h2 * a2 + h3 * a3 + term3 * term4

        # 更新粘弹性历史变量
        h1_new = h1 * a1 + m1 * term4
        h2_new = h2 * a2 + m2 * term4
        h3_new = h3 * a3 + m3 * term4

        # 塑性屈服判断
        stress_dev = self.deviatoric(stress_pred)
        back_stress_dev = self.deviatoric(back_stress)

        # 计算相对应力
        relative_stress = stress_dev - back_stress_dev

        # Von Mises 屈服准则
        J2 = self.calc_J2(relative_stress)
        sigma_eq = np.sqrt(3.0 * J2)

        # 当前屈服应力（考虑各向同性强化）
        sigma_y_current = sigma_y + H * epseq

        # 屈服函数
        phi = sigma_eq - sigma_y_current

        # 初始化塑性变量
        delta_lambda = 0.0
        dp = 0.0
        dplastic_strain = np.zeros(ntens, dtype=DTYPE)
        back_stress_new = back_stress.copy()
        plastic_strain_new = plastic_strain.copy()
        epseq_new = epseq

        # 如果发生屈服，进行塑性修正
        if phi > 0:
            # 流动方向
            if sigma_eq > 0:
                n = (3.0 / (2.0 * sigma_eq)) * relative_stress
            else:
                n = np.zeros(ntens, dtype=DTYPE)

            # 一致性条件求解塑性乘子
            denominator = 3.0 * mu0 * term3 + H

            if denominator > 0:
                delta_lambda = phi / denominator

                # delta_lambda = (sigma_eq - self.yield_stress) / denominator

                # 塑性应变增量
                dplastic_strain = delta_lambda * n

                # 更新等效塑性应变
                dp = delta_lambda
                epseq_new = epseq + dp

                # print(epseq_new, plastic_strain_new[0])

                # 更新塑性应变张量
                plastic_strain_new = plastic_strain + dplastic_strain

                # 更新背应力（随动强化）
                # Armstrong-Frederick 非线性随动强化模型
                back_stress_new = back_stress + H * n * dplastic_strain

        # 计算最终应力（考虑塑性修正）
        # 弹性应变增量 = 总应变增量 - 塑性应变增量
        dstrain_elastic = dstrain - dplastic_strain


        # 重新计算应力（使用弹性应变增量）
        term4_corrected = np.dot(Ce, dstrain_elastic)
        stress_final = np.dot(Ce, strain - plastic_strain_new) + h1_new * a1 + h2_new * a2 + h3_new * a3 + term3 * term4_corrected

        # 更新粘弹性历史变量（基于弹性应变增量）
        h1_final = h1 * a1 + m1 * term4_corrected
        h2_final = h2 * a2 + m2 * term4_corrected
        h3_final = h3 * a3 + m3 * term4_corrected

        # 计算一致切线刚度矩阵
        if phi <= 0:  # 弹性加载或卸载
            ddsdde = term3 * Ce
        else:  # 塑性加载
            # 弹性预测刚度
            C_el = term3 * Ce

            # 塑性修正项
            if sigma_eq > 0:
                # 投影张量
                n_tensor = np.outer(n, n)

                # 塑性模量
                H_eff = H

                # 弹塑性一致性切线
                scale = (3.0 * mu0 * term3) / (3.0 * mu0 * term3 + H_eff)
                C_ep = C_el - scale * np.dot(C_el, np.dot(n_tensor, C_el)) / (np.dot(n, np.dot(C_el, n)) + H_eff / 3.0)

                ddsdde = C_ep
            else:
                ddsdde = C_el

        ddsdde = term3 * Ce

        stress = stress_final
        state_variable_new['epseq'] = epseq_new
        state_variable_new['plastic_strain'] = plastic_strain_new
        state_variable_new['back_stress'] = back_stress_new
        state_variable_new['stress'] = stress
        state_variable_new['h1'] = h1_final
        state_variable_new['h2'] = h2_final
        state_variable_new['h3'] = h3_final

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

    def deviatoric(self, tensor):
        """计算张量的偏量部分"""
        if len(tensor) == 6:  # 三维情况
            trace = tensor[0] + tensor[1] + tensor[2]
            dev = tensor.copy()
            dev[0] -= trace / 3.0
            dev[1] -= trace / 3.0
            dev[2] -= trace / 3.0
            return dev
        elif len(tensor) == 3:  # 二维情况
            trace = tensor[0] + tensor[1]
            dev = tensor.copy()
            dev[0] -= trace / 2.0
            dev[1] -= trace / 2.0
            return dev
        else:
            return tensor

    def calc_J2(self, deviator):
        """计算第二应力不变量 J2"""
        if len(deviator) == 6:  # 三维情况
            J2 = 0.5 * (deviator[0] ** 2 + deviator[1] ** 2 + deviator[2] ** 2) + \
                 deviator[3] ** 2 + deviator[4] ** 2 + deviator[5] ** 2
        elif len(deviator) == 3:  # 二维情况
            J2 = 0.5 * (deviator[0] ** 2 + deviator[1] ** 2) + \
                 deviator[2] ** 2
        else:
            J2 = 0.0
        return J2
    

if __name__ == "__main__":
    from pyfem.utils.visualization import print_slots_dict

    print_slots_dict(ViscoElasticPlasticMaxwell.__slots_dict__)

    from pyfem.job.Job import Job

    job = Job(r'..\..\..\examples\mechanical\specimen3D\Job-1.toml')

    job.run()
