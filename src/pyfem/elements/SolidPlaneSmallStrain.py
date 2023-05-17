from numpy import empty, zeros, dot, ndarray

from pyfem.elements.BaseElement import BaseElement
from pyfem.elements.IsoElementShape import IsoElementShape
from pyfem.io.Dof import Dof
from pyfem.io.Material import Material
from pyfem.io.Section import Section
from pyfem.materials.BaseMaterial import BaseMaterial
from pyfem.utils.colors import error_style


class SolidPlaneSmallStrain(BaseElement):

    def __init__(self, element_id: int,
                 iso_element_shape: IsoElementShape,
                 connectivity: ndarray,
                 node_coords: ndarray,
                 section: Section,
                 dof: Dof,
                 material: Material,
                 material_data: BaseMaterial):

        super().__init__(element_id, iso_element_shape, connectivity, node_coords)
        self.dof = dof
        self.dof_names = ['u1', 'u2']
        if dof.names != self.dof_names:
            error_msg = f'{dof.names} is not the supported dof for {type(self).__name__} element'
            raise NotImplementedError(error_style(error_msg))
        self.element_dof_number = len(self.dof_names) * self.iso_element_shape.nodes_number
        self.material = material
        self.material_data = material_data
        self.section = section
        self.gp_b_matrices = empty(0)
        self.stiffness = empty(0)
        self.update_gp_b_matrices()
        self.update_stiffness()

    def update_gp_b_matrices(self):

        self.gp_b_matrices = zeros(shape=(self.iso_element_shape.gp_number, 3, self.element_dof_number))

        for igp, (gp_shape_gradient, gp_jacobi_inv) in \
                enumerate(zip(self.iso_element_shape.gp_shape_gradients, self.gp_jacobi_invs)):
            gp_dhdx = dot(gp_shape_gradient, gp_jacobi_inv)
            for i, val in enumerate(gp_dhdx):
                self.gp_b_matrices[igp, 0, i * 2] = val[0]
                self.gp_b_matrices[igp, 1, i * 2 + 1] = val[1]
                self.gp_b_matrices[igp, 2, i * 2] = val[1]
                self.gp_b_matrices[igp, 2, i * 2 + 1] = val[0]

    def update_stiffness(self):
        self.stiffness = zeros(shape=(self.element_dof_number, self.element_dof_number))

        ddsdde = self.material_data.ddsdde
        gp_weights = self.iso_element_shape.gp_weights
        gp_jacobi_dets = self.gp_jacobi_dets
        gp_b_matrices = self.gp_b_matrices
        gp_number = self.iso_element_shape.gp_number

        for i in range(gp_number):
            self.stiffness += dot(gp_b_matrices[i].transpose(), dot(ddsdde, gp_b_matrices[i])) * gp_weights[i] * \
                              gp_jacobi_dets[i]


if __name__ == "__main__":
    pass
