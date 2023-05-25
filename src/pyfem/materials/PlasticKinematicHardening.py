# -*- coding: utf-8 -*-
"""

"""
from typing import Optional, Dict

from numpy import array, zeros, ndarray, dot, sqrt, empty, ones

from pyfem.io.Material import Material
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
        self.ddsdde3d: ndarray = empty(0)
        self.tolerance: float = 1.0e-10
        self.create_tangent()

    def create_tangent(self):
        if self.option in self.allowed_option:
            if self.dimension == 3:
                self.option = None
            self.ddsdde = get_stiffness_from_young_poisson(self.dimension, self.young, self.poisson, self.option)
            self.ddsdde3d = get_stiffness_from_young_poisson(3, self.young, self.poisson, self.option)
        else:
            error_msg = f'{self.option} is not the allowed options {self.allowed_option}'
            raise NotImplementedError(error_style(error_msg))

    def get_tangent(self, state_variable: Dict[str, ndarray], dstate: ndarray) -> ndarray:
        # dstrain = dstate
        # ntens = len(dstrain)
        if state_variable == {}:
            state_variable['eelas'] = zeros(6)
            state_variable['eplas'] = zeros(6)
            state_variable['alpha'] = zeros(6)
            state_variable['sigma'] = zeros(6)

        # print(dstate)

        eelas = state_variable['eelas']
        eplas = state_variable['eplas']
        alpha = state_variable['alpha']
        sigma = state_variable['sigma']

        # print(state_variable)

        if len(dstate) == 6:
            dstrain = dstate
        else:
            dstrain = transform_2_to_3(dstate)

        eelas += dstrain

        ddsdde = self.ddsdde3d

        sigma += dot(ddsdde, dstrain)

        smises = get_smises(sigma - alpha)

        deqpl = 0.

        if smises > (1.0 + self.tolerance) * self.yield_stress:
            shydro = get_hydrostatic(sigma)

            flow = sigma - alpha

            flow[:3] = flow[:3] - shydro * ones(3)
            flow *= 1.0 / smises

            deqpl = (smises - self.yield_stress) / (self.eg3 + self.hard)
        
        #     alpha += self.hard * flow * deqpl
        #     eplas[:3] += 1.5 * flow[:3] * deqpl
        #     eelas[:3] += -1.5 * flow[:3] * deqpl
        #
        #     eplas[3:] += 3.0 * flow[3:] * deqpl
        #     eelas[3:] += -3.0 * flow[3:] * deqpl
        #
        #     sigma = alpha + flow * self.syield
        #     sigma[:3] += shydro * ones(3)
        #
        #     effg = self.eg * (self.syield + self.hard * deqpl) / smises
        #     effg2 = 2.0 * effg
        #     effg3 = 3.0 * effg
        #     efflam = 1.0 / 3.0 * (self.ebulk3 - effg2)
        #     effhdr = self.eg3 * self.hard / (self.eg3 + self.hard) - effg3
        #
        #     tang = zeros(shape=(6, 6))
        #     tang[:3, :3] = efflam
        #
        #     for i in range(3):
        #         tang[i, i] += effg2
        #         tang[i + 3, i + 3] += effg
        #
        #     tang += effhdr * outer(flow, flow)

        return self.ddsdde


def transform_2_to_3(s):
    return array([s[0], s[1], 0.0, s[2], 0.0, 0.0])


def transform_3_to_2(s, t):
    return array([s[0], s[1], s[3]]), array([(t[0, 0], t[0, 1], t[0, 3]), (t[1, 0], t[1, 1], t[1, 3]), (t[3, 0], t[3, 1], t[3, 3])])


def vonMisesStress(s):
    smises = 0.;

    if len(s) == 3:
        return sqrt(s[0] * s[0] + s[1] * s[1] - s[0] * s[1] + 3. * s[2] * s[2]);
    elif len(s) == 6:
        smises = (s[0] - s[1]) * (s[0] - s[1]) + \
                 (s[1] - s[2]) * (s[1] - s[2]) + \
                 (s[2] - s[0]) * (s[2] - s[0])

        smises += 6.0 * dot(s[3:], s[3:])
        return sqrt(0.5 * smises)


def get_smises(s: ndarray) -> ndarray:
    if len(s) == 3:
        return sqrt(s[0]**2 + s[1]**2 - s[0]*s[1] + 3*s[2]**2)
    elif len(s) == 6:
        smises = (s[0]-s[1])**2 + (s[1]-s[2])**2 + (s[2]-s[0])**2
        smises += 6 * sum([i**2 for i in s[3:]])
        return sqrt(0.5 * smises)


def get_hydrostatic(s: ndarray) -> ndarray:
    return sum(s[:3])/3.0


if __name__ == "__main__":
    from pyfem.Job import Job

    job = Job(r'F:\Github\pyfem\examples\rectangle\rectangle.toml')

    material_data = PlasticKinematicHardening(job.props.materials[0], 3)
    print(material_data.to_string())
