# -*- coding: utf-8 -*-
"""

"""
from copy import deepcopy

from numpy import all as np_all, diff as np_diff, abs as np_abs
from numpy import zeros, ndarray, dot, sqrt, outer, insert, delete, searchsorted, array

from pyfem.fem.Timer import Timer
from pyfem.fem.constants import DTYPE
from pyfem.io.Material import Material
from pyfem.io.Section import Section
from pyfem.materials.BaseMaterial import BaseMaterial
from pyfem.materials.ElasticIsotropic import get_stiffness_from_young_poisson
from pyfem.utils.colors import error_style


class User(BaseMaterial):
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
        'E': ('float', 'E'),
        'nu': ('float', 'nu'),
        'ycc': ('ndarray', '屈服数据'),
        'EBULK3': ('float', '3倍体积模量'),
        'EG': ('float', '剪切模量'),
        'EG2': ('float', '2倍剪切模量'),
        'EG3': ('float', '3倍剪切模量'),
        'ELAM': ('float', '拉梅常数'),
        'tolerance': ('float', '判断屈服的误差容限'),
    }

    __slots__ = BaseMaterial.__slots__ + [slot for slot in __slots_dict__.keys()]

    __data_keys__ = ['E', 'nu', 'ycc']

    def __init__(self, material: Material, dimension: int, section: Section) -> None:
        super().__init__(material, dimension, section)
        self.allowed_section_types = ('Volume', 'PlaneStress', 'PlaneStrain')

        self.data_keys = self.__data_keys__
        for i, key in enumerate(self.data_keys):
            self.data_dict[key] = material.data_dict[key]

        # self.data_keys = self.__data_keys__
        #
        # if len(self.material.data) != len(self.data_keys):
        #     raise NotImplementedError(error_style(self.get_data_length_error_msg()))
        # else:
        #     for i, key in enumerate(self.data_keys):
        #         self.data_dict[key] = material.data[i]

        self.E: float = self.data_dict['E']
        self.nu: float = self.data_dict['nu']
        # self.yield_stress: float = self.data_dict['Yield stress']
        self.ycc: ndarray = array(self.data_dict['ycc'])

        self.EBULK3: float = self.E / (1.0 - 2.0 * self.nu)
        self.EG2: float = self.E / (1.0 + self.nu)
        self.EG: float = self.EG2 / 2.0
        self.EG3: float = 3.0 * self.EG
        self.ELAM: float = (self.EBULK3 - self.EG2) / 3.0
        self.tolerance: float = 1.0e-9

        self.create_tangent()

    def create_tangent(self):
        if self.section.type in self.allowed_section_types:
            self.tangent = get_stiffness_from_young_poisson(self.dimension, self.E, self.nu, self.section.type)
        else:
            raise NotImplementedError(error_style(self.get_section_type_error_msg()))

    def get_tangent(self, variable: dict[str, ndarray],
                    state_variable: dict[str, ndarray],
                    state_variable_new: dict[str, ndarray],
                    element_id: int,
                    iqp: int,
                    ntens: int,
                    ndi: int,
                    nshr: int,
                    timer: Timer) -> tuple[ndarray, dict[str, ndarray]]:

        if state_variable == {} or timer.time0 == 0.0:
            state_variable['elastic_strain'] = zeros(ntens, dtype=DTYPE)
            state_variable['plastic_strain'] = zeros(ntens, dtype=DTYPE)
            state_variable['stress'] = zeros(ntens, dtype=DTYPE)
            state_variable['eq_plastic_strain'] = 0.0

        elastic_strain = deepcopy(state_variable['elastic_strain'])
        plastic_strain = deepcopy(state_variable['plastic_strain'])
        eq_plastic_strain = deepcopy(state_variable['eq_plastic_strain'])
        stress = deepcopy(state_variable['stress'])

        dstrain = variable['dstrain']

        E = self.E
        nu = self.nu
        newton_iteration = int(10)

        if self.section.type == 'PlaneStrain':
            dstrain = insert(dstrain, 2, 0)
        elif self.section.type == 'PlaneStress':
            dstrain = insert(dstrain, 2, -nu / (1 - nu) * (dstrain[0] + dstrain[1]))

        elastic_strain += dstrain

        ddsdde = zeros((ntens, ntens), dtype=DTYPE)

        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        mu = E / (2.0 * (1.0 + nu))

        ddsdde[:ndi, :ndi] += lam
        for i in range(ndi):
            ddsdde[i, i] += 2 * mu
        for i in range(ndi, ntens):
            ddsdde[i, i] += mu

        stress += dot(ddsdde, dstrain)

        smises = get_smises(stress)

        syield0, hard = uhard(eq_plastic_strain, self.ycc)

        if smises > (1.0 + self.tolerance) * syield0:
            hydrostatic_stress = sum(stress[:ndi]) / ndi
            # flow = deepcopy(stress)
            # flow[:ndi] = flow[:ndi] - hydrostatic_stress
            # flow *= 1.0 / smises
            # print('flow0', flow)

            deviatoric_stress = deepcopy(stress)
            deviatoric_stress[:ndi] = deviatoric_stress[:ndi] - hydrostatic_stress
            flow = deviatoric_stress / smises

            # print('flow1', flow)

            # 局部牛顿法求解等效冯mises应力和塑性应变增量
            syield = syield0
            dp = 0.0

            for iter_count in range(newton_iteration):
                rhs = smises - self.EG3 * dp - syield
                dp += rhs / (self.EG3 + hard)
                syield, hard = uhard(eq_plastic_strain + dp, self.ycc)
                if np_abs(rhs) < self.tolerance:
                    break
            else:
                print(f"警告：拉伸塑性算法在 {newton_iteration} 次迭代后没有收敛")

            stress[:ndi] = flow[:ndi] * syield + hydrostatic_stress
            plastic_strain[:ndi] += 1.5 * flow[:ndi] * dp
            elastic_strain[:ndi] -= 1.5 * flow[:ndi] * dp

            stress[ndi:ntens] = flow[ndi:ntens] * syield
            plastic_strain[ndi:ntens] += 3.0 * flow[ndi:ntens] * dp
            elastic_strain[ndi:ntens] -= 3.0 * flow[ndi:ntens] * dp

            eq_plastic_strain += dp

            plastic_dissipation = 0.5 * dp * (syield + syield0)

            EFFG = self.EG * syield / smises
            EFFG2 = 2.0 * EFFG
            EFFG3 = 3.0 / 2.0 * EFFG
            EFFLAM = (self.EBULK3 - EFFG2) / 3.0
            EFFHRD = self.EG3 * hard / (self.EG3 + hard) - EFFG3

            ddsdde = zeros(shape=(ntens, ntens), dtype=DTYPE)
            ddsdde[:ndi, :ndi] = EFFLAM
            for i in range(ndi):
                ddsdde[i, i] += EFFG2
            for i in range(ndi, ntens):
                ddsdde[i, i] += EFFG
            ddsdde += EFFHRD * outer(flow, flow)

        state_variable_new['elastic_strain'] = elastic_strain
        state_variable_new['plastic_strain'] = plastic_strain
        state_variable_new['eq_plastic_strain'] = eq_plastic_strain
        state_variable_new['stress'] = stress

        strain_energy = sum(plastic_strain * stress)

        if self.section.type == 'PlaneStrain':
            ddsdde = delete(delete(ddsdde, 2, axis=0), 2, axis=1)
            stress = delete(stress, 2)
        elif self.section.type == 'PlaneStress':
            ddsdde = delete(delete(ddsdde, 2, axis=0), 2, axis=1)
            ddsdde[0, 0] -= lam * lam / (lam + 2 * mu)
            ddsdde[0, 1] -= lam * lam / (lam + 2 * mu)
            ddsdde[1, 0] -= lam * lam / (lam + 2 * mu)
            ddsdde[1, 1] -= lam * lam / (lam + 2 * mu)
            stress = delete(stress, 2)

        output = {'stress': stress, 'strain_energy': strain_energy}

        return ddsdde, output


def get_smises(s: ndarray) -> float:
    if len(s) == 3:
        smises = sqrt(s[0] ** 2 + s[1] ** 2 - s[0] * s[1] + 3 * s[2] ** 2)
        return float(smises)
    elif len(s) >= 3:
        smises = (s[0] - s[1]) ** 2 + (s[1] - s[2]) ** 2 + (s[2] - s[0]) ** 2
        smises += 6 * sum([i ** 2 for i in s[3:]])
        smises = sqrt(0.5 * smises)
        return float(smises)
    else:
        raise NotImplementedError(error_style(f'unsupported stress dimension {len(s)}'))


def uhard(plas_eq, ycc):
    """
    计算当前屈服应力 (syield) 和硬化模量 (hard)

    参数:
    eqplas : float
        当前累积等效塑性应变
    table : ndarray, shape (2, n)
        第一行是屈服应力，第二行是对应的等效塑性应变
        table = np.array([
                        [200, 250, 300, 350],  # 屈服应力
                        [0.0, 0.01, 0.02, 0.03]  # 对应的等效塑性应变
                        ])
    返回:
    syield : float
        当前屈服应力
    hard : float
        当前硬化模量
    """
    # 获取表格中的屈服应力和应变数据
    syields = ycc[0, :]
    eqpls = ycc[1, :]

    # 确保 plas_eq 是一个单个值
    if isinstance(plas_eq, ndarray) and plas_eq.size > 1:
        raise ValueError("eqplas 应该是一个单个值，而不是数组")

    # 确保应变按升序排列
    if not np_all(np_diff(eqpls) > 0):
        raise ValueError("塑性应变数据必须按升序排列")

    # 使用 searchsorted 找到插入位置
    idx = searchsorted(eqpls, plas_eq)

    # 判断条件
    if idx == 0:
        s_yield = syields[0]
        hard = 0.0
    elif idx >= len(ycc):
        s_yield = syields[-1]
        hard = 0.0
    else:
        eqpl0, eqpl1 = eqpls[idx - 1:idx + 1]
        syiel0, syiel1 = syields[idx - 1:idx + 1]

        deqpl = eqpl1 - eqpl0
        if deqpl == 0:
            raise ValueError("塑性应变间隔为零，无法计算硬化模量")

        hard = (syiel1 - syiel0) / deqpl
        s_yield = syiel0 + (plas_eq - eqpl0) * hard

    return s_yield, hard


if __name__ == '__main__':
    import numpy as np

    print(uhard(0.005, np.array([[200, 250, 300, 350], [0.0, 0.01, 0.02, 0.03]])))
