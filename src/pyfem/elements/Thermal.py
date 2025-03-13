# -*- coding: utf-8 -*-
"""

"""
from numpy import array, zeros, dot, ndarray

from pyfem.elements.BaseElement import BaseElement
from pyfem.elements.set_element_field_variables import set_element_field_variables
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

    :ivar qp_temperatures: 积分点处的温度列表
    :vartype qp_temperatures: ndarray

    :ivar qp_heat_fluxes: 积分点处的热流密度列表
    :vartype qp_heat_fluxes: ndarray

    :ivar qp_ddsddts: 积分点处的材料热传导系数矩阵列表
    :vartype qp_ddsddts: list[ndarray]

    :ivar ntens: 总应力数量
    :vartype ntens: int

    :ivar ndi: 轴向应力数量
    :vartype ndi: int

    :ivar nshr: 剪切应力数量
    :vartype nshr: int
    """

    __slots_dict__: dict = {
        'qp_temperatures': ('ndarray', '积分点处的温度列表'),
        'qp_heat_fluxes': ('ndarray', '积分点处的热流密度列表'),
        'qp_ddsddts': ('list[ndarray]', '积分点处的材料热传导系数矩阵列表'),
        'ntens': ('int', '总应力数量'),
        'ndi': ('int', '轴向应力数量'),
        'nshr': ('int', '剪切应力数量')
    }

    __slots__: list = BaseElement.__slots__ + [slot for slot in __slots_dict__.keys()]

    __allowed_material_data_list__ = [('ThermalIsotropic', 'User')]

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

        self.allowed_material_data_list = self.__allowed_material_data_list__
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

        self.qp_temperatures: list[ndarray] = None  # type: ignore
        self.qp_heat_fluxes: list[ndarray] = None  # type: ignore
        self.qp_ddsddts: list[ndarray] = None  # type: ignore

    def update_element_material_stiffness_fint(self,
                                               is_update_material: bool = True,
                                               is_update_stiffness: bool = True,
                                               is_update_fint: bool = True, ) -> None:
        element_id = self.element_id
        timer = self.timer
        ntens = self.ntens
        ndi = self.ndi
        nshr = self.nshr

        qp_number = self.qp_number
        qp_weight_times_jacobi_dets = self.qp_weight_times_jacobi_dets
        qp_shape_values = self.iso_element_shape.qp_shape_values
        qp_dhdxex = self.qp_dhdxes

        qp_state_variables = self.qp_state_variables
        qp_state_variables_new = self.qp_state_variables_new

        element_dof_values = self.element_dof_values
        element_ddof_values = self.element_ddof_values

        material_data = self.material_data_list[0]

        if is_update_stiffness:
            self.element_stiffness = zeros(shape=(self.element_dof_number, self.element_dof_number), dtype=DTYPE)

        if is_update_fint:
            self.element_fint = zeros(self.element_dof_number, dtype=DTYPE)

        if is_update_material:
            self.qp_ddsddts = list()
            self.qp_temperatures = list()
            self.qp_heat_fluxes = list()

        for i in range(qp_number):
            if is_update_material:
                qp_weight_times_jacobi_det = qp_weight_times_jacobi_dets[i]
                qp_shape_value = qp_shape_values[i]
                qp_dhdx = qp_dhdxex[i]
                qp_temperature = dot(qp_shape_value, element_dof_values)
                qp_dtemperature = dot(qp_shape_value, element_ddof_values)
                qp_temperature_gradient = dot(qp_dhdx, element_dof_values)
                qp_dtemperature_gradient = dot(qp_dhdx, element_ddof_values)

                variable = {'temperature': qp_temperature,
                            'dtemperature': qp_dtemperature,
                            'temperature_gradient': qp_temperature_gradient,
                            'dtemperature_gradient': qp_dtemperature_gradient}
                qp_ddsddt, qp_output = material_data.get_tangent(variable=variable,
                                                                 state_variable=qp_state_variables[i],
                                                                 state_variable_new=qp_state_variables_new[i],
                                                                 element_id=element_id,
                                                                 iqp=i,
                                                                 ntens=ntens,
                                                                 ndi=ndi,
                                                                 nshr=nshr,
                                                                 timer=timer)
                qp_heat_flux = qp_output['heat_flux']
                self.qp_ddsddts.append(qp_ddsddt)
                self.qp_temperatures.append(qp_temperature)
                self.qp_heat_fluxes.append(qp_heat_flux)
            else:
                qp_weight_times_jacobi_det = qp_weight_times_jacobi_dets[i]
                qp_ddsddt = self.qp_ddsddts[i]
                qp_heat_flux = self.qp_heat_fluxes[i]

            if is_update_stiffness:
                self.element_stiffness += dot(qp_dhdx.transpose(), dot(qp_ddsddt, qp_dhdx)) * qp_weight_times_jacobi_det

            if is_update_fint:
                self.element_fint += dot(qp_dhdx.transpose(), qp_heat_flux) * qp_weight_times_jacobi_det

    def update_element_field_variables(self) -> None:
        self.qp_field_variables['heat_flux'] = array(self.qp_heat_fluxes, dtype=DTYPE)
        for key in self.qp_state_variables_new[0].keys():
            if key not in ['heat_flux']:
                variable = []
                for qp_state_variable_new in self.qp_state_variables_new:
                    variable.append(qp_state_variable_new[key])
                self.qp_field_variables[f'SDV-{key}'] = array(variable, dtype=DTYPE)
        self.element_nodal_field_variables = set_element_field_variables(self.qp_field_variables, self.iso_element_shape, self.dimension)


if __name__ == "__main__":
    from pyfem.utils.visualization import print_slots_dict

    print_slots_dict(Thermal.__slots_dict__)

    from pyfem.Job import Job

    job = Job(r'F:\Github\pyfem\examples\thermal\1element\hex8\Job-1.toml')

    job.run()
