# -*- coding: utf-8 -*-
"""

"""
from typing import Optional, Dict

from numpy import array, zeros, diag, float64, ndarray

from pyfem.io.Material import Material
from pyfem.materials.BaseMaterial import BaseMaterial
from pyfem.materials.ElasticIsotropic import get_stiffness_from_young_poisson
from pyfem.utils.colors import error_style


class PlasticKinematicHardening(BaseMaterial):
    allowed_option = ['PlaneStress', 'PlaneStrain', None]

    def __init__(self, material: Material, dimension: int, option: Optional[str] = None) -> None:
        super().__init__(material, dimension, option)
        self.create_tangent()

    def create_tangent(self):
        young = self.material.data[0]
        poisson = self.material.data[1]

        if self.option in self.allowed_option:
            if self.dimension == 3:
                self.option = None
            self.ddsdde = get_stiffness_from_young_poisson(self.dimension, young, poisson, self.option)
        else:
            error_msg = f'{self.option} is not the allowed options {self.allowed_option}'
            raise NotImplementedError(error_style(error_msg))

    def get_tangent(self, state_variable: Dict[str, ndarray], ddof: ndarray) -> ndarray:
        if state_variable == {}:
            state_variable['eelas'] = zeros(6)
            state_variable['eplas'] = zeros(6)
            state_variable['alpha'] = zeros(6)
            state_variable['sigma'] = zeros(6)

        eelas = state_variable['eelas']
        eplas = state_variable['eplas']
        alpha = state_variable['alpha']
        sigma = state_variable['sigma']

        # print(state_variable)
        # if len(kinematics.dstrain) == 6:
        #     dstrain = kinematics.dstrain
        # else:
        #     dstrain = transform2To3(kinematics.dstrain)
        #
        # eelas += dstrain
        #
        # sigma += dot(self.ctang, dstrain)
        #
        # tang = self.ctang
        #
        # smises = vonMisesStress(sigma - alpha)
        #
        # deqpl = 0.
        #
        # if smises > (1.0 + self.tolerance) * self.syield:
        #     shydro = hydrostaticStress(sigma)
        #
        #     flow = sigma - alpha
        #
        #     flow[:3] = flow[:3] - shydro * ones(3)
        #     flow *= 1.0 / smises
        #
        #     deqpl = (smises - self.syield) / (self.eg3 + self.hard)
        #
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


if __name__ == "__main__":
    from pyfem.Job import Job

    job = Job(r'F:\Github\pyfem\examples\rectangle\rectangle.toml')

    material_data = PlasticKinematicHardening(job.props.materials[0], 3)
    print(material_data.to_string())
