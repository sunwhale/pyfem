# -*- coding: utf-8 -*-
"""

"""
from numpy import array, zeros, dot, ndarray, average, outer

from pyfem.elements.BaseElement import BaseElement
from pyfem.fem.Timer import Timer
from pyfem.fem.constants import DTYPE
from pyfem.io.Dof import Dof
from pyfem.io.Material import Material
from pyfem.io.Section import Section
from pyfem.isoelements.IsoElementShape import IsoElementShape
from pyfem.materials.get_material_data import MaterialData
from pyfem.utils.colors import error_style


class Diffusion(BaseElement):
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
        'qp_concentrations': ('ndarray', '积分点处的浓度列表'),
        'qp_dconcentrations': ('ndarray', '积分点处的浓度增量列表'),
        'qp_concentration_fluxes': ('ndarray', '积分点处的浓度通量列表'),
        'qp_ddsddcs': ('list[ndarray]', '积分点处的材料扩散系数矩阵列表'),
        'ntens': ('int', '总应力数量'),
        'ndi': ('int', '轴向应力数量'),
        'nshr': ('int', '剪切应力数量'),
    }

    __slots__: list = BaseElement.__slots__ + [slot for slot in __slots_dict__.keys()]

    __allowed_material_data_list__ = [('DiffusionIsotropic', 'User')]

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

        self.dof_names = ['C']
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
        self.element_ftime = zeros(element_dof_number, dtype=DTYPE)
        self.element_stiffness = zeros(shape=(self.element_dof_number, self.element_dof_number), dtype=DTYPE)

        self.qp_concentrations: list[ndarray] = None  # type: ignore
        self.qp_dconcentrations: list[ndarray] = None  # type: ignore
        self.qp_concentration_fluxes: list[ndarray] = None  # type: ignore
        self.qp_ddsddcs: list[ndarray] = None  # type: ignore

    def update_element_material_stiffness_fint(self,
                                               is_update_material: bool = True,
                                               is_update_stiffness: bool = True,
                                               is_update_fint: bool = True, ) -> None:
        element_id = self.element_id
        timer = self.timer
        ntens = self.ntens
        ndi = self.ndi
        nshr = self.nshr
        dtime = timer.dtime

        qp_number = self.qp_number
        qp_weight_times_jacobi_dets = self.qp_weight_times_jacobi_dets
        qp_shape_values = self.iso_element_shape.qp_shape_values
        qp_shape_gradients = self.iso_element_shape.qp_shape_gradients
        qp_dhdxex = self.qp_dhdxes

        qp_state_variables = self.qp_state_variables
        qp_state_variables_new = self.qp_state_variables_new

        element_dof_values = self.element_dof_values
        element_ddof_values = self.element_ddof_values

        material_data = self.material_data_list[0]

        if is_update_stiffness:
            self.element_stiffness = zeros(shape=(self.element_dof_number, self.element_dof_number), dtype=DTYPE)
            self.element_ftime = zeros(self.element_dof_number, dtype=DTYPE)

        if is_update_fint:
            self.element_fint = zeros(self.element_dof_number, dtype=DTYPE)
            self.element_ftime = zeros(self.element_dof_number, dtype=DTYPE)

        if is_update_material:
            self.qp_ddsddcs = list()
            self.qp_concentrations = list()
            self.qp_dconcentrations = list()
            self.qp_concentration_fluxes = list()

        for i in range(qp_number):
            if is_update_material:
                qp_weight_times_jacobi_det = qp_weight_times_jacobi_dets[i]
                qp_shape_value = qp_shape_values[i]
                qp_shape_gradient = qp_shape_gradients[i]
                qp_dhdx = qp_dhdxex[i]
                qp_concentration = dot(qp_shape_value, element_dof_values)
                qp_dconcentration = dot(qp_shape_value, element_ddof_values)
                qp_concentration_gradient = dot(qp_dhdx, element_dof_values)
                qp_dconcentration_gradient = dot(qp_dhdx, element_ddof_values)

                variable = {'concentration': qp_concentration,
                            'dconcentration': qp_dconcentration,
                            'concentration_gradient': qp_concentration_gradient,
                            'dconcentration_gradient': qp_dconcentration_gradient}
                qp_ddsddc, qp_output = material_data.get_tangent(variable=variable,
                                                                 state_variable=qp_state_variables[i],
                                                                 state_variable_new=qp_state_variables_new[i],
                                                                 element_id=element_id,
                                                                 iqp=i,
                                                                 ntens=ntens,
                                                                 ndi=ndi,
                                                                 nshr=nshr,
                                                                 timer=timer)
                qp_concentration_flux = qp_output['concentration_flux']
                self.qp_ddsddcs.append(qp_ddsddc)
                self.qp_concentrations.append(qp_concentration)
                self.qp_dconcentrations.append(qp_dconcentration)
                self.qp_concentration_fluxes.append(qp_concentration_flux)
            else:
                qp_shape_value = qp_shape_values[i]
                qp_dhdx = qp_dhdxex[i]
                qp_weight_times_jacobi_det = qp_weight_times_jacobi_dets[i]
                qp_ddsddc = self.qp_ddsddcs[i]
                qp_concentration = self.qp_concentrations[i]
                qp_dconcentration = self.qp_dconcentrations[i]
                qp_concentration_flux = self.qp_concentration_fluxes[i]

            if is_update_stiffness:
                # self.element_stiffness += 1.0 / dtime * outer(qp_shape_value, qp_shape_value) * qp_weight_times_jacobi_det + \
                #                           dot(qp_dhdx.transpose(), dot(qp_ddsddc, qp_dhdx)) * qp_weight_times_jacobi_det
                self.element_stiffness += 1.0 / dtime * outer(qp_shape_value, qp_shape_value) * qp_weight_times_jacobi_det
                self.element_stiffness += dot(qp_dhdx.transpose(), dot(qp_ddsddc, qp_dhdx)) * qp_weight_times_jacobi_det

                self.element_ftime += qp_concentration / dtime * qp_shape_value * qp_weight_times_jacobi_det

            if is_update_fint:
                # self.element_fint += 1.0 / dtime * qp_shape_value * (qp_concentration + qp_dconcentration) * qp_weight_times_jacobi_det + \
                #                      dot(qp_dhdx.transpose(), qp_concentration_flux) * qp_weight_times_jacobi_det

                # self.element_ftime += qp_concentration / dtime * qp_shape_value * qp_weight_times_jacobi_det

                self.element_fint += 1.0 / dtime * qp_shape_value * (qp_concentration + qp_dconcentration) * qp_weight_times_jacobi_det
                self.element_fint += dot(qp_dhdx.transpose(), qp_concentration_flux) * qp_weight_times_jacobi_det

    def update_element_field_variables(self) -> None:
        qp_concentrations = self.qp_concentrations
        qp_concentration_fluxes = self.qp_concentration_fluxes

        average_temperatures = average(qp_concentrations, axis=0)
        average_heat_fluxes = average(qp_concentration_fluxes, axis=0)

        self.qp_field_variables['concentration'] = array(qp_concentrations, dtype=DTYPE)
        self.qp_field_variables['concentration_flux'] = array(qp_concentration_fluxes, dtype=DTYPE)

        self.element_average_field_variables['C'] = average_temperatures
        if len(average_heat_fluxes) >= 1:
            self.element_average_field_variables['CFL1'] = average_heat_fluxes[0]
        if len(average_heat_fluxes) >= 2:
            self.element_average_field_variables['CFL2'] = average_heat_fluxes[1]
        if len(average_heat_fluxes) >= 3:
            self.element_average_field_variables['CFL3'] = average_heat_fluxes[2]


if __name__ == "__main__":
    from pyfem.utils.visualization import print_slots_dict

    print_slots_dict(Diffusion.__slots_dict__)

    from pyfem.Job import Job

    job = Job(r'F:\Github\pyfem\examples\diffusion\rectangle\Job-1.toml')

    job.run()
