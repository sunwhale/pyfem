from numpy import array, empty, zeros, dot, ndarray, average

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
                 material_data: BaseMaterial) -> None:

        super().__init__(element_id, iso_element_shape, connectivity, node_coords)
        self.dof = dof
        self.dof_names = ['u1', 'u2']
        self.field_variable_dict = {
            'strain': ['E11', 'E22', 'E12'],
            'stress': ['S11', 'S22', 'S12']
        }
        if dof.names != self.dof_names:
            error_msg = f'{dof.names} is not the supported dof of {type(self).__name__} element'
            raise NotImplementedError(error_style(error_msg))
        self.element_dof_number = len(self.dof_names) * self.iso_element_shape.nodes_number
        self.material = material
        self.section = section
        self.material_data = material_data
        self.gp_b_matrices = empty(0)
        self.stiffness = empty(0)
        self.update_gp_b_matrices()
        self.update_stiffness()

    def update_gp_b_matrices(self) -> None:

        self.gp_b_matrices = zeros(shape=(self.iso_element_shape.gp_number, 3, self.element_dof_number))

        for igp, (gp_shape_gradient, gp_jacobi_inv) in \
                enumerate(zip(self.iso_element_shape.gp_shape_gradients, self.gp_jacobi_invs)):
            gp_dhdx = dot(gp_shape_gradient, gp_jacobi_inv)
            for i, val in enumerate(gp_dhdx):
                self.gp_b_matrices[igp, 0, i * 2] = val[0]
                self.gp_b_matrices[igp, 1, i * 2 + 1] = val[1]
                self.gp_b_matrices[igp, 2, i * 2] = val[1]
                self.gp_b_matrices[igp, 2, i * 2 + 1] = val[0]

    def update_stiffness(self) -> None:
        self.stiffness = zeros(shape=(self.element_dof_number, self.element_dof_number))

        ddsdde = self.material_data.get_tangent()
        gp_weights = self.iso_element_shape.gp_weights
        gp_jacobi_dets = self.gp_jacobi_dets
        gp_b_matrices = self.gp_b_matrices
        gp_number = self.iso_element_shape.gp_number

        for i in range(gp_number):
            self.stiffness += dot(gp_b_matrices[i].transpose(), dot(ddsdde, gp_b_matrices[i])) * gp_weights[i] * \
                              gp_jacobi_dets[i]

    def update_field_variables(self, solution: ndarray) -> None:
        gp_b_matrices = self.gp_b_matrices
        gp_number = self.iso_element_shape.gp_number
        ddsdde = self.material_data.get_tangent()

        self.element_dof_values = solution[self.element_dof_ids]

        gp_strains = []
        gp_stresses = []
        for i in range(gp_number):
            gp_strain = dot(gp_b_matrices[i], self.element_dof_values)
            gp_stress = dot(ddsdde, gp_strain)
            gp_strains.append(gp_strain)
            gp_stresses.append(gp_stress)

        self.gp_field_variables['strain'] = array(gp_strains)
        self.gp_field_variables['stress'] = array(gp_stresses)

        # self.average_field_variables['strain'] = average(self.gp_field_variables['strain'], axis=0)
        # self.average_field_variables['stress'] = average(self.gp_field_variables['stress'], axis=0)

        average_strain = average(self.gp_field_variables['strain'], axis=0)
        average_stress = average(self.gp_field_variables['stress'], axis=0)

        self.average_field_variables['E11'] = average_strain[0]
        self.average_field_variables['E22'] = average_strain[1]
        self.average_field_variables['E12'] = average_strain[2]
        self.average_field_variables['S11'] = average_stress[0]
        self.average_field_variables['S22'] = average_stress[1]
        self.average_field_variables['S12'] = average_stress[2]


if __name__ == "__main__":
    pass
