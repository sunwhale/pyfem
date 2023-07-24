# -*- coding: utf-8 -*-
"""

"""
from numpy import array, zeros, dot, ndarray, average

from pyfem.elements.BaseElement import BaseElement
from pyfem.fem.Timer import Timer
from pyfem.fem.constants import DTYPE
from pyfem.io.Dof import Dof
from pyfem.io.Material import Material
from pyfem.io.Section import Section
from pyfem.isoelements.IsoElementShape import IsoElementShape
from pyfem.materials.get_material_data import MaterialData
from pyfem.utils.colors import error_style


class Thermal(BaseElement):
    """
    温度单元。

    :ivar gp_temperatures: 积分点处的温度列表
    :vartype gp_temperatures: ndarray

    :ivar gp_heat_fluxes: 积分点处的热流密度列表
    :vartype gp_heat_fluxes: ndarray

    :ivar gp_ddsddts: 积分点处的材料热传导系数矩阵列表
    :vartype gp_ddsddts: list[ndarray]

    :ivar ntens: 总应力数量
    :vartype ntens: int

    :ivar ndi: 轴向应力数量
    :vartype ndi: int

    :ivar nshr: 剪切应力数量
    :vartype nshr: int
    """

    __slots_dict__: dict = {
        'gp_temperatures': ('ndarray', '积分点处的温度列表'),
        'gp_heat_fluxes': ('ndarray', '积分点处的热流密度列表'),
        'gp_ddsddts': ('list[ndarray]', '积分点处的材料热传导系数矩阵列表'),
        'ntens': ('int', '总应力数量'),
        'ndi': ('int', '轴向应力数量'),
        'nshr': ('int', '剪切应力数量')
    }

    __slots__: list = BaseElement.__slots__ + [slot for slot in __slots_dict__.keys()]

    def __init__(self, element_id: int,
                 iso_element_shape: IsoElementShape,
                 connectivity: ndarray,
                 node_coords: ndarray,
                 dof: Dof,
                 materials: list[Material],
                 section: Section,
                 material_data_list: list[MaterialData],
                 timer: Timer) -> None:

        super().__init__(element_id, iso_element_shape, connectivity, node_coords)

        self.allowed_material_data_list = [('ThermalIsotropic',)]
        self.allowed_material_number = 1

        self.dof = dof
        self.materials = materials
        self.section = section
        self.material_data_list = material_data_list
        self.check_materials()
        self.timer = timer

        self.dof_names = ['T']
        self.ntens = 6
        self.ndi = 3
        self.nshr = 3

        if dof.names != self.dof_names:
            error_msg = f'{dof.names} is not the supported dof of {type(self).__name__} element'
            raise NotImplementedError(error_style(error_msg))

        element_dof_number = len(self.dof_names) * self.iso_element_shape.nodes_number

        self.element_dof_number = element_dof_number
        self.element_dof_values = zeros(element_dof_number, dtype=DTYPE)
        self.element_ddof_values = zeros(element_dof_number, dtype=DTYPE)
        self.element_fint = zeros(element_dof_number, dtype=DTYPE)
        self.element_stiffness = zeros(shape=(self.element_dof_number, self.element_dof_number), dtype=DTYPE)

        self.gp_temperatures: list[ndarray] = None  # type: ignore
        self.gp_heat_fluxes: list[ndarray] = None  # type: ignore
        self.gp_ddsddts: list[ndarray] = None  # type: ignore

    def update_element_material_stiffness_fint(self,
                                               is_update_material: bool = True,
                                               is_update_stiffness: bool = True,
                                               is_update_fint: bool = True, ) -> None:
        element_id = self.element_id
        timer = self.timer
        ntens = self.ntens
        ndi = self.ndi
        nshr = self.nshr

        gp_number = self.gp_number
        gp_weight_times_jacobi_dets = self.gp_weight_times_jacobi_dets
        gp_shape_values = self.iso_element_shape.gp_shape_values
        gp_shape_gradients = self.iso_element_shape.gp_shape_gradients

        gp_state_variables = self.gp_state_variables
        gp_state_variables_new = self.gp_state_variables_new

        element_dof_values = self.element_dof_values
        element_ddof_values = self.element_ddof_values

        material_data = self.material_data_list[0]

        if is_update_stiffness:
            self.element_stiffness = zeros(shape=(self.element_dof_number, self.element_dof_number), dtype=DTYPE)

        if is_update_fint:
            self.element_fint = zeros(self.element_dof_number, dtype=DTYPE)

        if is_update_material:
            self.gp_ddsddts = list()
            self.gp_temperatures = list()
            self.gp_heat_fluxes = list()

        for i in range(gp_number):
            if is_update_material:
                gp_weight_times_jacobi_det = gp_weight_times_jacobi_dets[i]
                gp_shape_value = gp_shape_values[i]
                gp_shape_gradient = gp_shape_gradients[i]
                gp_temperature = dot(gp_shape_value, element_dof_values)
                gp_dtemperature = dot(gp_shape_value, element_ddof_values)
                gp_temperature_gradient = dot(gp_shape_gradient, element_dof_values)
                gp_dtemperature_gradient = dot(gp_shape_gradient, element_ddof_values)

                variable = {'temperature': gp_temperature,
                            'dtemperature': gp_dtemperature,
                            'temperature_gradient': gp_temperature_gradient,
                            'dtemperature_gradient': gp_dtemperature_gradient}
                gp_ddsddt, gp_output = material_data.get_tangent(variable=variable,
                                                                 state_variable=gp_state_variables[i],
                                                                 state_variable_new=gp_state_variables_new[i],
                                                                 element_id=element_id,
                                                                 igp=i,
                                                                 ntens=ntens,
                                                                 ndi=ndi,
                                                                 nshr=nshr,
                                                                 timer=timer)
                gp_heat_flux = gp_output['heat_flux']
                self.gp_ddsddts.append(gp_ddsddt)
                self.gp_temperatures.append(gp_temperature)
                self.gp_heat_fluxes.append(gp_heat_flux)
            else:
                gp_shape_gradient = gp_shape_gradients[i]
                gp_weight_times_jacobi_det = gp_weight_times_jacobi_dets[i]
                gp_ddsddt = self.gp_ddsddts[i]
                gp_heat_flux = self.gp_heat_fluxes[i]

            if is_update_stiffness:
                self.element_stiffness += dot(gp_shape_gradient.transpose(), dot(gp_ddsddt, gp_shape_gradient)) * \
                                          gp_weight_times_jacobi_det

            if is_update_fint:
                self.element_fint += dot(gp_shape_gradient.transpose(), gp_heat_flux) * gp_weight_times_jacobi_det

    def update_element_field_variables(self) -> None:
        gp_temperatures = self.gp_temperatures
        gp_heat_fluxes = self.gp_heat_fluxes

        average_temperatures = average(gp_temperatures, axis=0)
        average_heat_fluxes = average(gp_heat_fluxes, axis=0)

        self.gp_field_variables['temperature'] = array(gp_temperatures, dtype=DTYPE)
        self.gp_field_variables['heat_flux'] = array(gp_heat_fluxes, dtype=DTYPE)

        self.element_average_field_variables['T'] = average_temperatures
        if len(average_heat_fluxes) >= 1:
            self.element_average_field_variables['HFL1'] = average_heat_fluxes[0]
        if len(average_heat_fluxes) >= 2:
            self.element_average_field_variables['HFL2'] = average_heat_fluxes[1]
        if len(average_heat_fluxes) >= 3:
            self.element_average_field_variables['HFL3'] = average_heat_fluxes[2]


if __name__ == "__main__":
    from pyfem.utils.visualization import print_slots_dict

    print_slots_dict(Thermal.__slots_dict__)

    from pyfem.Job import Job

    job = Job(r'F:\Github\pyfem\examples\thermal\1element\hex8\Job-1.toml')

    job.run()
