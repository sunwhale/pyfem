# -*- coding: utf-8 -*-
"""

"""
from typing import List

from numpy import array, zeros, dot, ndarray, average

from pyfem.elements.BaseElement import BaseElement
from pyfem.elements.IsoElementShape import IsoElementShape
from pyfem.fem.Timer import Timer
from pyfem.fem.constants import DTYPE
from pyfem.io.Dof import Dof
from pyfem.io.Material import Material
from pyfem.io.Section import Section
from pyfem.materials.BaseMaterial import BaseMaterial
from pyfem.utils.colors import error_style


class ThermalVolume(BaseElement):
    __slots__ = BaseElement.__slots__ + ('gp_temperatures', 'gp_heat_fluxes')

    def __init__(self, element_id: int,
                 iso_element_shape: IsoElementShape,
                 connectivity: ndarray,
                 node_coords: ndarray,
                 dof: Dof,
                 material: Material,
                 section: Section,
                 material_data: BaseMaterial,
                 timer: Timer) -> None:

        super().__init__(element_id, iso_element_shape, connectivity, node_coords)

        self.dof = dof
        self.material = material
        self.section = section
        self.material_data = material_data
        self.timer = timer

        self.dof_names = ['T']
        if dof.names != self.dof_names:
            error_msg = f'{dof.names} is not the supported dof of {type(self).__name__} element'
            raise NotImplementedError(error_style(error_msg))

        element_dof_number = len(self.dof_names) * self.iso_element_shape.nodes_number

        self.element_dof_number = element_dof_number
        self.element_dof_values = zeros(element_dof_number, dtype=DTYPE)
        self.element_ddof_values = zeros(element_dof_number, dtype=DTYPE)
        self.element_fint = zeros(element_dof_number, dtype=DTYPE)
        self.element_stiffness = None  # type: ignore

    def update_element_material_stiffness_fint(self) -> None:
        element_id = self.element_id
        timer = self.timer

        gp_number = self.gp_number
        gp_weight_times_jacobi_dets = self.gp_weight_times_jacobi_dets
        gp_shape_values = self.iso_element_shape.gp_shape_values
        gp_shape_gradients = self.iso_element_shape.gp_shape_gradients

        gp_state_variables = self.gp_state_variables
        gp_state_variables_new = self.gp_state_variables_new

        element_dof_values = self.element_dof_values
        element_ddof_values = self.element_ddof_values

        self.element_stiffness = zeros(shape=(self.element_dof_number, self.element_dof_number), dtype=DTYPE)
        self.element_fint = zeros(self.element_dof_number, dtype=DTYPE)

        gp_ddsddes = []
        gp_temperatures = []
        gp_heat_fluxes = []

        for i in range(gp_number):
            gp_weight_times_jacobi_det = gp_weight_times_jacobi_dets[i]
            gp_shape_value = gp_shape_values[i]
            gp_shape_gradient = gp_shape_gradients[i]
            gp_temperature = dot(gp_shape_value, element_dof_values)
            gp_dtemperature = dot(gp_shape_value, element_ddof_values)
            variable = {'temperature': gp_temperature, 'dtemperature': gp_dtemperature}
            gp_ddsdde, gp_output = self.material_data.get_tangent(variable=variable,
                                                                  state_variable=gp_state_variables[i],
                                                                  state_variable_new=gp_state_variables_new[i],
                                                                  element_id=element_id,
                                                                  igp=i,
                                                                  ntens=6,
                                                                  ndi=3,
                                                                  nshr=3,
                                                                  timer=timer)
            gp_heat_flux = gp_output['heat_flux']

            self.element_stiffness += dot(gp_shape_gradient, dot(gp_ddsdde, gp_shape_gradient.transpose())) * \
                                      gp_weight_times_jacobi_det

            self.element_fint += dot(gp_shape_gradient, gp_temperature) * gp_weight_times_jacobi_det

            gp_ddsddes.append(gp_ddsdde)
            gp_temperatures.append(gp_temperature)
            gp_heat_fluxes.append(gp_heat_flux)

        self.gp_ddsddes = gp_ddsddes
        self.gp_temperatures = gp_temperatures
        self.gp_heat_fluxes = gp_heat_fluxes

    def update_element_stiffness(self) -> None:
        self.element_stiffness = zeros(shape=(self.element_dof_number, self.element_dof_number), dtype=DTYPE)

        gp_weight_times_jacobi_dets = self.gp_weight_times_jacobi_dets
        gp_b_matrices = self.gp_b_matrices
        gp_b_matrices_transpose = self.gp_b_matrices_transpose
        gp_number = self.gp_number
        gp_ddsddes = self.gp_ddsddes

        for i in range(gp_number):
            self.element_stiffness += dot(gp_b_matrices_transpose[i], dot(gp_ddsddes[i], gp_b_matrices[i])) * \
                                      gp_weight_times_jacobi_dets[i]

    def update_element_fint(self) -> None:
        gp_weight_times_jacobi_dets = self.gp_weight_times_jacobi_dets
        gp_number = self.gp_number
        gp_shape_gradients = self.iso_element_shape.gp_shape_gradients
        gp_heat_fluxes = self.gp_heat_fluxes

        self.element_fint = zeros(self.element_dof_number, dtype=DTYPE)
        # for i in range(gp_number):
        #     self.element_fint += dot(gp_shape_gradients[i], gp_heat_fluxes[i]) * gp_weight_times_jacobi_dets[i]

    def update_element_field_variables(self) -> None:
        gp_temperatures = self.gp_temperatures
        gp_heat_fluxes = self.gp_heat_fluxes

        average_temperatures = average(gp_temperatures, axis=0)
        average_heat_fluxes = average(gp_heat_fluxes, axis=0)

        print(average_heat_fluxes.shape)

        self.gp_field_variables['temperature'] = array(gp_temperatures, dtype=DTYPE)
        self.gp_field_variables['heat_flux'] = array(gp_heat_fluxes, dtype=DTYPE)

        self.element_average_field_variables['T'] = average_temperatures
        self.element_average_field_variables['HFL1'] = average_heat_fluxes[0]
        self.element_average_field_variables['HFL2'] = average_heat_fluxes[1]


if __name__ == "__main__":
    from pyfem.Job import Job
    job = Job(r'F:\Github\pyfem\examples\thermal\1element\hex8\Job-1.toml')

    # job.assembly.element_data_list[0].show()

    job.run()
