from numpy import dot, ndarray

from pyfem.io.Material import Material
from pyfem.materials.BaseMaterial import BaseMaterial
from pyfem.materials.isotropic_elastic_stiffness import get_stiffness_from_young_poisson


class PlaneStress(BaseMaterial):

    def __init__(self, material: Material):
        super().__init__()
        self.material = material

        young = self.material.data[0]
        poisson = self.material.data[1]
        self.ddsdde = get_stiffness_from_young_poisson(2, young, poisson, 'stress')

    def get_stress(self, strain: ndarray) -> ndarray:
        sigma = dot(self.ddsdde, strain)
        return sigma

    def get_tangent(self) -> ndarray:
        return self.ddsdde


if __name__ == "__main__":
    from pyfem.io.Properties import Properties

    props = Properties()
    props.read_file(r'F:\Github\pyfem\examples\rectangle\rectangle.toml')

    plane_stress = PlaneStress(props.materials[0])
    print(props.materials[0].to_string())
    print(plane_stress.to_string())
