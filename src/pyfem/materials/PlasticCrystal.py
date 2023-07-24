# -*- coding: utf-8 -*-
"""

"""
from copy import deepcopy

from numpy import zeros, ndarray, dot, sqrt, outer, insert, delete, ones

from pyfem.fem.Timer import Timer
from pyfem.fem.constants import DTYPE
from pyfem.io.Material import Material
from pyfem.io.Section import Section
from pyfem.materials.BaseMaterial import BaseMaterial
from pyfem.materials.ElasticIsotropic import get_stiffness_from_young_poisson
from pyfem.utils.colors import error_style


class PlasticCrystal(BaseMaterial):
    r"""
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

    :ivar total_number_of_slips: 总的滑移系数量
    :vartype total_number_of_slips: int

    :ivar h_matrix: 硬化系数矩阵
    :vartype h_matrix: ndarray

    .. math::
        \frac{{0.5{h_{1,1}}}}{{{{\left( {{h_{1,1}}\left( {\rho _{di}^{(1)} + \rho _m^{(1)}} \right) + {h_{1,2}}\left( {\rho _{di}^{(2)} + \rho _m^{(2)}} \right) + {h_{1,3}}\left( {\rho _{di}^{(3)} + \rho _m^{(3)}} \right) + {h_{1,4}}\left( {\rho _{di}^{(4)} + \rho _m^{(4)}} \right) + {h_{1,5}}\left( {\rho _{di}^{(5)} + \rho _m^{(5)}} \right) + {h_{1,6}}\left( {\rho _{di}^{(6)} + \rho _m^{(6)}} \right)} \right)}^{0.5}}}}

    """

    __slots_dict__: dict = {
        'E': ('float', 'Young\'s modulus E'),
        'nu': ('float', 'Poisson\'s ratio nu'),
        'yield_stress': ('float', 'Yield stress'),
        'hard': ('float', 'Hardening coefficient'),
        'EBULK3': ('float', '3倍体积模量'),
        'EG': ('float', '剪切模量'),
        'EG2': ('float', '2倍剪切模量'),
        'EG3': ('float', '3倍剪切模量'),
        'ELAM': ('float', '拉梅常数'),
        'tolerance': ('float', '判断屈服的误差容限'),
        'total_number_of_slips': ('int', '总的滑移系数量'),
        'h_matrix': ('ndarray', '硬化系数矩阵')
    }

    __slots__ = BaseMaterial.__slots__ + [slot for slot in __slots_dict__.keys()]

    def __init__(self, material: Material, dimension: int, section: Section) -> None:
        super().__init__(material, dimension, section)
        self.allowed_section_types = ('Volume', 'PlaneStress', 'PlaneStrain')

        self.data_keys = ['Young\'s modulus E', 'Poisson\'s ratio nu', 'Yield stress', 'Hardening coefficient']

        if len(self.material.data) != len(self.data_keys):
            raise NotImplementedError(error_style(self.get_data_length_error_msg()))
        else:
            for i, key in enumerate(self.data_keys):
                self.data_dict[key] = material.data[i]

        self.E: float = self.data_dict['Young\'s modulus E']
        self.nu: float = self.data_dict['Poisson\'s ratio nu']
        self.yield_stress: float = self.data_dict['Yield stress']
        self.hard: float = self.data_dict['Hardening coefficient']

        self.EBULK3: float = self.E / (1.0 - 2.0 * self.nu)
        self.EG2: float = self.E / (1.0 + self.nu)
        self.EG: float = self.EG2 / 2.0
        self.EG3: float = 3.0 * self.EG
        self.ELAM: float = (self.EBULK3 - self.EG2) / 3.0
        self.tolerance: float = 1.0e-10

        self.total_number_of_slips: int = 12
        self.h_matrix: ndarray = ones((self.total_number_of_slips, self.total_number_of_slips))

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
                    igp: int,
                    ntens: int,
                    ndi: int,
                    nshr: int,
                    timer: Timer) -> tuple[ndarray, dict[str, ndarray]]:

        strain = variable['strain']
        dstrain = variable['dstrain']

        if state_variable == {}:
            state_variable['rho_m'] = zeros(self.total_number_of_slips, dtype=DTYPE)
            state_variable['rho_di'] = zeros(self.total_number_of_slips, dtype=DTYPE)
            state_variable['m_e'] = zeros((self.total_number_of_slips, 3), dtype=DTYPE)
            state_variable['n_e'] = zeros((self.total_number_of_slips, 3), dtype=DTYPE)
            state_variable['gamma'] = zeros(self.total_number_of_slips, dtype=DTYPE)

        rho_m = deepcopy(state_variable['rho_m'])
        rho_di = deepcopy(state_variable['rho_di'])
        m_e = deepcopy(state_variable['m_e'])
        n_e = deepcopy(state_variable['n_e'])
        gamma = deepcopy(state_variable['gamma'])

        rho_m = rho_m + sum(dstrain)

        ddsdde = zeros((ntens, ntens), dtype=DTYPE)
        E = self.E
        nu = self.nu
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        mu = E / (2.0 * (1.0 + nu))

        if element_id == 0 and igp == 0:
            print(E, nu)
            print(rho_m)

        ddsdde[:ndi, :ndi] += lam
        for i in range(ndi):
            ddsdde[i, i] += 2 * mu
        for i in range(ndi, ntens):
            ddsdde[i, i] += mu

        # dstrain = variable['dstrain']

        stress = ddsdde.dot(strain + dstrain) + sum(rho_m) * 100

        state_variable_new['rho_m'] = rho_m
        state_variable_new['rho_di'] = rho_di
        state_variable_new['m_e'] = m_e
        state_variable_new['n_e'] = n_e
        state_variable_new['gamma'] = gamma
        output = {'stress': stress}

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


if __name__ == "__main__":
    # from pyfem.utils.visualization import print_slots_dict
    #
    # print_slots_dict(PlasticCrystal.__slots_dict__)

    from pyfem.Job import Job

    job = Job(r'..\..\..\examples\mechanical\1element\hex8_crystal\Job-1.toml')

    # job.assembly.element_data_list[0].material_data_list[0].show()

    print(job.assembly.element_data_list[0].gp_state_variables[0]['n_e'])

    # job.run()
