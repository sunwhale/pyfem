from typing import Optional, Tuple

from numpy import array, outer, diag, float64, ndarray

from pyfem.io.Material import Material
from pyfem.materials.BaseMaterial import BaseMaterial
from pyfem.utils.colors import error_style


class ElasticIsotropic(BaseMaterial):
    allowed_option = ['PlaneStress', 'PlaneStrain', None]

    def __init__(self, material: Material, dimension: int, option: Optional[str] = None) -> None:
        super().__init__(material, dimension, option)
        self.create_tangent()

    def create_tangent(self):
        young = self.material.data[0]
        poisson = self.material.data[1]

        if self.option in self.allowed_option:
            if self.dimension == 3:
                self.option = None
            self.ddsdde = get_stiffness_from_young_poisson(self.dimension, young, poisson, self.option)
        else:
            error_msg = f'{self.option} is not the allowed options {self.allowed_option}'
            raise NotImplementedError(error_style(error_msg))


def get_lame_from_young_poisson(young: float, poisson: float, plane: Optional[str]) -> Tuple[float, float]:
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
    mu = young / (2.0 * (1.0 + poisson))
    lam = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))

    if plane == 'PlaneStress':
        lam = 2 * lam * mu / (lam + 2 * mu)

    return lam, mu


def get_stiffness_from_lame(dim: int, lam: float, mu: float) -> ndarray:
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
    sym = (dim + 1) * dim // 2
    o = array([1.] * dim + [0.] * (sym - dim), dtype=float64)
    oot = outer(o, o)
    do1 = diag(o + 1.0)

    lam_array = array(lam)[..., None, None]
    mu_array = array(mu)[..., None, None]
    return lam_array * oot + mu_array * do1


def get_stiffness_from_young_poisson(dim: int, young: float, poisson: float, plane: Optional[str]) -> ndarray:
    """
    Compute stiffness tensor corresponding to Young's modulus and Poisson's
    ratio.
    """

    lam, mu = get_lame_from_young_poisson(young, poisson, plane)

    return get_stiffness_from_lame(dim, lam, mu)


def get_stiffness_from_lame_mixed(dim: int, lam: float, mu: float) -> ndarray:
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

    return get_stiffness_from_lame(dim, lam, mu)


def get_stiffness_from_young_poisson_mixed(dim: int, young: float, poisson: float, plane) -> ndarray:
    """
    Compute stiffness tensor corresponding to Young's modulus and Poisson's
    ratio for mixed formulation.
    """
    lam, mu = get_lame_from_young_poisson(young, poisson, plane)

    return get_stiffness_from_lame_mixed(dim, lam, mu)


def get_bulk_from_lame(lam: float, mu: float) -> float:
    r"""
    Compute bulk modulus from Lamé parameters.

    .. math::
        \gamma = \lambda + {2 \over 3} \mu
    """
    return lam + 2.0 * mu / 3.0


def get_bulk_from_young_poisson(young: float, poisson: float, plane: Optional[str]) -> float:
    """
    Compute bulk modulus corresponding to Young's modulus and Poisson's ratio.
    """
    lam, mu = get_lame_from_young_poisson(young, poisson, plane)

    return get_bulk_from_lame(lam, mu)


def get_lame_from_stiffness(stiffness: ndarray, plane: Optional[str]) -> Tuple[float, float]:
    """
    Compute Lamé parameters from an isotropic stiffness tensor.
    """
    lam = float(stiffness[..., 0, 1])
    mu = float(stiffness[..., -1, -1])
    if plane == 'PlaneStress':
        lam = - 2.0 * mu * lam / (lam - 2.0 * mu)

    return lam, mu


def get_young_poisson_from_stiffness(stiffness: ndarray, plane: Optional[str]) -> Tuple[float, float]:
    """
    Compute Young's modulus and Poisson's ratio from an isotropic stiffness
    tensor.
    """
    lam, mu = get_lame_from_stiffness(stiffness, plane)
    young = (3 * lam * mu + 2 * mu ** 2) / (lam + mu)
    poisson = lam / (2 * lam + 2 * mu)

    return young, poisson


if __name__ == "__main__":
    from pyfem.io.Properties import Properties

    props = Properties()
    props.read_file(r'F:\Github\pyfem\examples\rectangle\rectangle.toml')

    material_data = ElasticIsotropic(props.materials[0], 3)
    print(material_data.to_string())
