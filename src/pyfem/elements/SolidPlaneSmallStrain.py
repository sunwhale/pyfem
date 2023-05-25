# -*- coding: utf-8 -*-
"""

"""
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
        if dof.names != self.dof_names:
            error_msg = f'{dof.names} is not the supported dof of {type(self).__name__} element'
            raise NotImplementedError(error_style(error_msg))
        element_dof_number = len(self.dof_names) * self.iso_element_shape.nodes_number
        self.element_dof_number = element_dof_number
        self.element_dof_values = zeros(element_dof_number)
        self.element_ddof_values = zeros(element_dof_number)
        self.element_fint = zeros(element_dof_number)
        self.material = material
        self.section = section
        self.material_data = material_data
        self.gp_b_matrices = empty(0)
        self.element_stiffness = empty(0)
        self.create_gp_b_matrices()

    def create_gp_b_matrices(self) -> None:
        self.gp_b_matrices = zeros(shape=(self.iso_element_shape.gp_number, 3, self.element_dof_number))

        for igp, (gp_shape_gradient, gp_jacobi_inv) in \
                enumerate(zip(self.iso_element_shape.gp_shape_gradients, self.gp_jacobi_invs)):
            gp_dhdx = dot(gp_shape_gradient, gp_jacobi_inv)
            for i, val in enumerate(gp_dhdx):
                self.gp_b_matrices[igp, 0, i * 2] = val[0]
                self.gp_b_matrices[igp, 1, i * 2 + 1] = val[1]
                self.gp_b_matrices[igp, 2, i * 2] = val[1]
                self.gp_b_matrices[igp, 2, i * 2 + 1] = val[0]

    def update_element_dof_values(self, global_dof_values: ndarray) -> None:
        old_element_dof_values = self.element_dof_values
        self.element_dof_values = global_dof_values[self.element_dof_ids]
        self.element_ddof_values = self.element_dof_values - old_element_dof_values

    def update_material_state(self) -> None:
        gp_number = self.iso_element_shape.gp_number
        gp_b_matrices = self.gp_b_matrices
        gp_state_variables = self.gp_state_variables
        element_dof_values = self.element_dof_values
        element_ddof_values = self.element_ddof_values

        gp_ddsddes = []

        for i in range(gp_number):
            gp_strain = dot(gp_b_matrices[i], element_dof_values)
            gp_dstrain = dot(gp_b_matrices[i], element_ddof_values)
            gp_ddsdde = self.material_data.get_tangent(state_variable=gp_state_variables[i],
                                                       state=gp_strain,
                                                       dstate=gp_dstrain,
                                                       time=1.0,
                                                       dtime=1.0)
            gp_ddsddes.append(gp_ddsdde)
        self.gp_ddsddes = array(gp_ddsddes)

    def update_element_stiffness(self) -> None:
        self.element_stiffness = zeros(shape=(self.element_dof_number, self.element_dof_number))

        gp_weights = self.iso_element_shape.gp_weights
        gp_jacobi_dets = self.gp_jacobi_dets
        gp_b_matrices = self.gp_b_matrices
        gp_number = self.iso_element_shape.gp_number
        gp_ddsddes = self.gp_ddsddes

        for i in range(gp_number):
            self.element_stiffness += dot(gp_b_matrices[i].transpose(), dot(gp_ddsddes[i], gp_b_matrices[i])) * gp_weights[i] * \
                              gp_jacobi_dets[i]

    def update_element_fint(self) -> None:
        # gp_stresses = []
        # for gp_state_variable in self.gp_state_variables:
        #     gp_stresses.append(gp_state_variable['stress'])
        # print(gp_stresses)

        gp_b_matrices = self.gp_b_matrices
        gp_number = self.iso_element_shape.gp_number
        gp_ddsddes = self.gp_ddsddes

        gp_strains = []
        gp_stresses = []
        for i in range(gp_number):
            ddsdde = gp_ddsddes[i]
            gp_strain = dot(gp_b_matrices[i], self.element_dof_values)
            gp_stress = dot(ddsdde, gp_strain)
            gp_strains.append(gp_strain)
            gp_stresses.append(gp_stress)

        print()



        self.element_fint = dot(self.element_stiffness, self.element_dof_values)

    def update_element_field_variables(self) -> None:
        gp_b_matrices = self.gp_b_matrices
        gp_number = self.iso_element_shape.gp_number
        gp_ddsddes = self.gp_ddsddes

        gp_strains = []
        gp_stresses = []
        for i in range(gp_number):
            ddsdde = gp_ddsddes[i]
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
