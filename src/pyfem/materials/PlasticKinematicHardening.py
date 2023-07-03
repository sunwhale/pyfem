# -*- coding: utf-8 -*-
"""

"""
from copy import deepcopy
from typing import Dict, Tuple

from numpy import zeros, ndarray, dot, sqrt, outer, insert, delete

from pyfem.fem.Timer import Timer
from pyfem.fem.constants import DTYPE
from pyfem.io.Material import Material
from pyfem.io.Section import Section
from pyfem.materials.BaseMaterial import BaseMaterial
from pyfem.materials.ElasticIsotropic import get_stiffness_from_young_poisson
from pyfem.utils.colors import error_style


class PlasticKinematicHardening(BaseMaterial):

    def __init__(self, material: Material, dimension: int, section: Section) -> None:
        super().__init__(material, dimension, section)
        self.allowed_section_types = ('PlaneStress', 'PlaneStrain', 'Volume')
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
        if self.section.type in self.allowed_section_types:
            self.ddsdde = get_stiffness_from_young_poisson(self.dimension, self.young, self.poisson, self.section.type)
        else:
            error_msg = f'{self.section.type} is not the allowed section types {self.allowed_section_types} of the material {type(self).__name__}, please check the definition of the section {self.section.name}'
            raise NotImplementedError(error_style(error_msg))

    def get_tangent(self, variable: Dict[str, ndarray],
                    state_variable: Dict[str, ndarray],
                    state_variable_new: Dict[str, ndarray],
                    element_id: int,
                    igp: int,
                    ntens: int,
                    ndi: int,
                    nshr: int,
                    timer: Timer) -> Tuple[ndarray, Dict[str, ndarray]]:

        if state_variable == {}:
            state_variable['elastic_strain'] = zeros(ntens, dtype=DTYPE)
            state_variable['plastic_strain'] = zeros(ntens, dtype=DTYPE)
            state_variable['back_stress'] = zeros(ntens, dtype=DTYPE)
            state_variable['stress'] = zeros(ntens, dtype=DTYPE)

        elastic_strain = deepcopy(state_variable['elastic_strain'])
        plastic_strain = deepcopy(state_variable['plastic_strain'])
        back_stress = deepcopy(state_variable['back_stress'])
        stress = deepcopy(state_variable['stress'])

        dstrain = variable['dstrain']

        E = self.young
        nu = self.poisson

        if self.section.type == 'PlaneStrain':
            dstrain = insert(dstrain, 2, 0)
        elif self.section.type == 'PlaneStress':
            dstrain = insert(dstrain, 2, -nu / (1 - nu) * (dstrain[0] + dstrain[1]))

        elastic_strain += dstrain

        ddsdde = zeros((ntens, ntens), dtype=DTYPE)

        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        mu = E / (2.0 * (1.0 + nu))

        ddsdde[:ndi, :ndi] += lam
        for i in range(ndi):
            ddsdde[i, i] += 2 * mu
        for i in range(ndi, ntens):
            ddsdde[i, i] += mu

        stress += dot(ddsdde, dstrain)
        s = stress - back_stress
        smises = get_smises(s)

        # if element_id == 0 and igp == 0:
        #     print(ddsdde)
        #     print(dstrain)
        #     print(stress)
        #     print(dot(ddsdde, dstrain))

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

            # if element_id == 0 and igp == 0:
            #     print(stress)

            EFFG = self.EG * (self.yield_stress + self.hard * delta_p) / smises
            EFFG2 = 2.0 * EFFG
            EFFG3 = 3.0 * EFFG
            EFFLAM = (self.EBULK3 - EFFG2) / 3.0
            EFFHRD = self.EG3 * self.hard / (self.EG3 + self.hard) - EFFG3

            ddsdde = zeros(shape=(ntens, ntens), dtype=DTYPE)
            ddsdde[:ndi, :ndi] = EFFLAM

            for i in range(ndi):
                ddsdde[i, i] += EFFG2

            for i in range(ndi, ntens):
                ddsdde[i, i] += EFFG

            ddsdde += EFFHRD * outer(flow, flow)

        # if element_id == 0 and igp == 0:
        #     print(stress)

        state_variable_new['elastic_strain'] = elastic_strain
        state_variable_new['plastic_strain'] = plastic_strain
        state_variable_new['back_stress'] = back_stress
        state_variable_new['stress'] = stress

        if self.section.type == 'PlaneStrain':
            ddsdde = delete(delete(ddsdde, 2, axis=0), 2, axis=1)
            stress = delete(stress, 2)
        elif self.section.type == 'PlaneStress':
            ddsdde = delete(delete(ddsdde, 2, axis=0), 2, axis=1)
            ddsdde[0, 0] -= lam * lam / (lam + 2 * mu)
            ddsdde[0, 1] -= lam * lam / (lam + 2 * mu)
            ddsdde[1, 0] -= lam * lam / (lam + 2 * mu)
            ddsdde[1, 1] -= lam * lam / (lam + 2 * mu)
            stress = delete(stress, 2)

        output = {'stress': stress}

        return ddsdde, output


def get_smises(s: ndarray) -> float:
    if len(s) == 3:
        smises = sqrt(s[0] ** 2 + s[1] ** 2 - s[0] * s[1] + 3 * s[2] ** 2)
        return float(smises)
    elif len(s) >= 3:
        smises = (s[0] - s[1]) ** 2 + (s[1] - s[2]) ** 2 + (s[2] - s[0]) ** 2
        smises += 6 * sum([i ** 2 for i in s[3:]])
        smises = sqrt(0.5 * smises)
        return float(smises)
    else:
        raise NotImplementedError(error_style(f'unsupported stress dimension {len(s)}'))


if __name__ == "__main__":
    from pyfem.Job import Job

    job = Job(r'F:\Github\pyfem\examples\rectangle\rectangle.toml')

    material_data = PlasticKinematicHardening(job.props.materials[0], 3, job.props.sections[0])
