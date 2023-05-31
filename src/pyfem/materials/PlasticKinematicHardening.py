# -*- coding: utf-8 -*-
"""

"""
from copy import deepcopy
from typing import Optional, Dict, Tuple

from numpy import zeros, ndarray, dot, sqrt, outer

from pyfem.io.Material import Material
from pyfem.fem.Timer import Timer
from pyfem.materials.BaseMaterial import BaseMaterial
from pyfem.materials.ElasticIsotropic import get_stiffness_from_young_poisson
from pyfem.utils.colors import error_style


class PlasticKinematicHardening(BaseMaterial):
    allowed_option = ['PlaneStress', 'PlaneStrain', None]

    def __init__(self, material: Material, dimension: int, option: Optional[str] = None) -> None:
        super().__init__(material, dimension, option)
        self.young: float = self.material.data[0]
        self.poisson: float = self.material.data[1]
        self.yield_stress: float = self.material.data[2]
        self.hard: float = self.material.data[3]
        self.EBULK3: float = self.young / (1.0 - 2.0 * self.poisson)
        self.EG2: float = self.young / (1.0 + self.poisson)
        self.EG: float = self.EG2 / 2.0
        self.EG3: float = 3.0 * self.EG
        self.ELAM: float = (self.EBULK3 - self.EG2) / 3.0
        self.tolerance: float = 1.0e-10
        self.create_tangent()

    def create_tangent(self):
        if self.option in self.allowed_option:
            if self.dimension == 3:
                self.option = None
            self.ddsdde = get_stiffness_from_young_poisson(self.dimension, self.young, self.poisson, self.option)
        else:
            error_msg = f'{self.option} is not the allowed options {self.allowed_option}'
            raise NotImplementedError(error_style(error_msg))

    def get_tangent(self, state_variable: Dict[str, ndarray],
                    state_variable_new: Dict[str, ndarray],
                    state: ndarray,
                    dstate: ndarray,
                    element_id: int,
                    igp: int,
                    ntens: int,
                    ndi: int,
                    nshr: int,
                    timer: Timer) -> Tuple[ndarray, ndarray]:

        if state_variable == {}:
            state_variable['elastic_strain'] = zeros(ntens)
            state_variable['plastic_strain'] = zeros(ntens)
            state_variable['back_stress'] = zeros(ntens)
            state_variable['stress'] = zeros(ntens)

        elastic_strain = deepcopy(state_variable['elastic_strain'])
        plastic_strain = deepcopy(state_variable['plastic_strain'])
        back_stress = deepcopy(state_variable['back_stress'])
        stress = deepcopy(state_variable['stress'])

        dstrain = dstate
        elastic_strain += dstrain
        ddsdde = self.ddsdde
        stress += dot(ddsdde, dstrain)
        smises = get_smises(stress - back_stress)

        if smises > (1.0 + self.tolerance) * self.yield_stress:
            hydrostatic_stress = sum(stress[:ndi]) / 3.0
            flow = stress - back_stress
            flow[:ndi] = flow[:ndi] - hydrostatic_stress
            flow *= 1.0 / smises

            delta_p = (smises - self.yield_stress) / (self.EG3 + self.hard)
            back_stress += self.hard * flow * delta_p

            plastic_strain[:ndi] += 1.5 * flow[:ndi] * delta_p
            elastic_strain[:ndi] -= 1.5 * flow[:ndi] * delta_p

            plastic_strain[ndi:] += 3.0 * flow[ndi:] * delta_p
            elastic_strain[ndi:] -= 3.0 * flow[ndi:] * delta_p

            stress = back_stress + flow * self.yield_stress
            stress[:ndi] += hydrostatic_stress

            EFFG = self.EG * (self.yield_stress + self.hard * delta_p) / smises
            EFFG2 = 2.0 * EFFG
            EFFG3 = 3.0 * EFFG
            EFFLAM = (self.EBULK3 - EFFG2) / 3.0
            EFFHRD = self.EG3 * self.hard / (self.EG3 + self.hard) - EFFG3

            ddsdde = zeros(shape=(ntens, ntens))
            ddsdde[:ndi, :ndi] = EFFLAM

            for i in range(ndi):
                ddsdde[i, i] += EFFG2

            for i in range(ndi, ntens):
                ddsdde[i, i] += EFFG

            ddsdde += EFFHRD * outer(flow, flow)

        state_variable_new['elastic_strain'] = elastic_strain
        state_variable_new['plastic_strain'] = plastic_strain
        state_variable_new['back_stress'] = back_stress
        state_variable_new['stress'] = stress

        return ddsdde, stress


def get_smises(s: ndarray) -> float:
    if len(s) == 3:
        smises = sqrt(s[0] ** 2 + s[1] ** 2 - s[0] * s[1] + 3 * s[2] ** 2)
        return float(smises)
    elif len(s) == 6:
        smises = (s[0] - s[1]) ** 2 + (s[1] - s[2]) ** 2 + (s[2] - s[0]) ** 2
        smises += 6 * sum([i ** 2 for i in s[3:]])
        smises = sqrt(0.5 * smises)
        return float(smises)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    from pyfem.Job import Job

    job = Job(r'F:\Github\pyfem\examples\rectangle\rectangle.toml')

    material_data = PlasticKinematicHardening(job.props.materials[0], 3)
    print(material_data.to_string())
