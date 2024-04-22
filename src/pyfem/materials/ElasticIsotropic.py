# -*- coding: utf-8 -*-
"""

"""
from copy import deepcopy

from numpy import array, ndarray, dot, zeros

from pyfem.fem.Timer import Timer
from pyfem.fem.constants import DTYPE
from pyfem.io.Material import Material
from pyfem.io.Section import Section
from pyfem.materials.BaseMaterial import BaseMaterial
from pyfem.utils.colors import error_style


class ElasticIsotropic(BaseMaterial):
    """
    各项同性弹性材料。

    支持的截面属性：('Volume', 'PlaneStress', 'PlaneStrain')

    :ivar E: Young's modulus E
    :vartype E: float

    :ivar nu: Poisson's ratio nu
    :vartype nu: float
    """

    __slots_dict__: dict = {
        'E': ('float', 'Young\'s modulus E'),
        'nu': ('float', 'Poisson\'s ratio nu'),
    }

    __slots__ = BaseMaterial.__slots__ + [slot for slot in __slots_dict__.keys()]

    def __init__(self, material: Material, dimension: int, section: Section) -> None:
        super().__init__(material, dimension, section)
        self.allowed_section_types = ('Volume', 'PlaneStress', 'PlaneStrain')

        self.data_keys = ['Young\'s modulus E', 'Poisson\'s ratio nu']

        if len(self.material.data) != len(self.data_keys):
            raise NotImplementedError(error_style(self.get_data_length_error_msg()))
        else:
            for i, key in enumerate(self.data_keys):
                self.data_dict[key] = material.data[i]

        self.E: float = self.data_dict['Young\'s modulus E']
        self.nu: float = self.data_dict['Poisson\'s ratio nu']

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

        # 全量格式
        # strain = variable['strain']
        # stress = dot(self.tangent, strain)

        # 增量格式
        strain = variable['strain']
        dstrain = variable['dstrain']
        if state_variable == {} or timer.time0 == 0.0:
            state_variable['stress'] = zeros(len(strain), dtype=DTYPE)
        stress = deepcopy(state_variable['stress'])
        stress += dot(self.tangent, dstrain)
        state_variable_new['stress'] = stress
        
        strain_energy = 0.5 * sum(stress * strain)
        output = {'stress': stress, 'strain_energy': strain_energy}
        return self.tangent, output


def get_lame_from_young_poisson(E: float, nu: float, plane: str) -> tuple[float, float]:
    r"""
    Compute Lamé parameters from Young's modulus and Poisson's ratio.

    The relationship between Lamé parameters and Young's modulus, Poisson's
    ratio (see [1],[2]):

    .. math::
        \lambda = {\nu E \over (1+\nu)(1-2\nu)},\qquad \mu = {E \over 2(1+\nu)}

    The plain stress hypothesis:

    .. math::
       \bar\lambda = {2\lambda\mu \over \lambda + 2\mu}

    [1] I.S. Sokolnikoff: Mathematical Theory of Elasticity. New York, 1956.

    [2] T.J.R. Hughes: The Finite Element Method, Linear Static and Dynamic
    Finite Element Analysis. New Jersey, 1987.
    """
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    if plane == 'PlaneStress':
        lam = 2 * lam * mu / (lam + 2 * mu)

    return lam, mu


def get_stiffness_from_lame(dimension: int, lam: float, mu: float) -> ndarray:
    r"""
    Compute stiffness tensor corresponding to Lamé parameters.

    .. math::
        {\bf D}_{(2D)} = \begin{bmatrix} \lambda + 2\mu & \lambda & 0\\
        \lambda & \lambda + 2\mu & 0\\ 0 & 0 & \mu \end{bmatrix}

    .. math::
        {\bf D}_{(3D)} = \begin{bmatrix} \lambda + 2\mu & \lambda &
        \lambda & 0 & 0 & 0\\ \lambda & \lambda + 2\mu & \lambda & 0 & 0 & 0 \\
        \lambda & \lambda & \lambda + 2\mu & 0 & 0 & 0 \\ 0 & 0 & 0 & \mu & 0 &
        0 \\ 0 & 0 & 0 & 0 & \mu & 0 \\ 0 & 0 & 0 & 0 & 0 & \mu\\ \end{bmatrix}
    """

    # sym = (dimension + 1) * dimension // 2
    # o = array([1.] * dimension + [0.] * (sym - dimension), dtype=DTYPE)
    # oot = outer(o, o)
    # do1 = diag(o + 1.0)
    #
    # lam_array = array(lam, dtype=DTYPE)[..., None, None]
    # mu_array = array(mu, dtype=DTYPE)[..., None, None]
    #
    # D = lam_array * oot + mu_array * do1

    assert dimension == 2 or dimension == 3, "Invalid dimension. Must be 2 or 3."

    if dimension == 2:
        D = array([[lam + 2 * mu, lam, 0],
                   [lam, lam + 2 * mu, 0],
                   [0, 0, mu]])
    else:
        D = array([[lam + 2 * mu, lam, lam, 0, 0, 0],
                   [lam, lam + 2 * mu, lam, 0, 0, 0],
                   [lam, lam, lam + 2 * mu, 0, 0, 0],
                   [0, 0, 0, mu, 0, 0],
                   [0, 0, 0, 0, mu, 0],
                   [0, 0, 0, 0, 0, mu]])

    return D


def get_stiffness_from_young_poisson(dimension: int, E: float, nu: float, plane: str) -> ndarray:
    """
    Compute stiffness tensor corresponding to Young's modulus and Poisson's
    ratio.
    """

    lam, mu = get_lame_from_young_poisson(E, nu, plane)

    return get_stiffness_from_lame(dimension, lam, mu)


def get_stiffness_from_lame_mixed(dimension: int, lam: float, mu: float) -> ndarray:
    r"""
    Compute stiffness tensor corresponding to Lamé parameters for mixed
    formulation.

    .. math::
        {\bf D}_{(2D)} = \begin{bmatrix} \widetilde\lambda + 2\mu &
        \widetilde\lambda & 0\\ \widetilde\lambda & \widetilde\lambda + 2\mu &
        0\\ 0 & 0 & \mu \end{bmatrix}

    .. math::
        {\bf D}_{(3D)} = \begin{bmatrix} \widetilde\lambda + 2\mu &
        \widetilde\lambda & \widetilde\lambda & 0 & 0 & 0\\ \widetilde\lambda &
        \widetilde\lambda + 2\mu & \widetilde\lambda & 0 & 0 & 0 \\
        \widetilde\lambda & \widetilde\lambda & \widetilde\lambda + 2\mu & 0 &
        0 & 0 \\ 0 & 0 & 0 & \mu & 0 & 0 \\ 0 & 0 & 0 & 0 & \mu & 0 \\ 0 & 0 &
        0 & 0 & 0 & \mu\\ \end{bmatrix}

    where

    .. math::
       \widetilde\lambda = -{2\over 3} \mu
    """
    lam = - 2.0 / 3.0 * mu

    return get_stiffness_from_lame(dimension, lam, mu)


def get_stiffness_from_young_poisson_mixed(dimension: int, E: float, nu: float, plane) -> ndarray:
    """
    Compute stiffness tensor corresponding to Young's modulus and Poisson's
    ratio for mixed formulation.
    """
    lam, mu = get_lame_from_young_poisson(E, nu, plane)

    return get_stiffness_from_lame_mixed(dimension, lam, mu)


def get_bulk_from_lame(lam: float, mu: float) -> float:
    r"""
    Compute bulk modulus from Lamé parameters.

    .. math::
        \gamma = \lambda + {2 \over 3} \mu
    """
    return lam + 2.0 * mu / 3.0


def get_bulk_from_young_poisson(E: float, nu: float, plane: str) -> float:
    """
    Compute bulk modulus corresponding to Young's modulus and Poisson's ratio.
    """
    lam, mu = get_lame_from_young_poisson(E, nu, plane)

    return get_bulk_from_lame(lam, mu)


def get_lame_from_stiffness(stiffness: ndarray, plane: str) -> tuple[float, float]:
    """
    Compute Lamé parameters from an isotropic stiffness tensor.
    """
    lam = float(stiffness[..., 0, 1])
    mu = float(stiffness[..., -1, -1])
    if plane == 'PlaneStress':
        lam = - 2.0 * mu * lam / (lam - 2.0 * mu)

    return lam, mu


def get_young_poisson_from_stiffness(stiffness: ndarray, plane: str) -> tuple[float, float]:
    """
    Compute Young's modulus and Poisson's ratio from an isotropic stiffness
    tensor.
    """
    lam, mu = get_lame_from_stiffness(stiffness, plane)
    E = (3 * lam * mu + 2 * mu ** 2) / (lam + mu)
    nu = lam / (2 * lam + 2 * mu)

    return E, nu


if __name__ == "__main__":
    from pyfem.utils.visualization import print_slots_dict

    print_slots_dict(ElasticIsotropic.__slots_dict__)

    from pyfem.io.Properties import Properties

    props = Properties()
    props.read_file(r'..\..\..\examples\mechanical\plane\Job-1.toml')
    material_data = ElasticIsotropic(props.materials[1], 3, props.sections[0])
    material_data.show()
