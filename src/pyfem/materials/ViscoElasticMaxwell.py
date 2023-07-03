# -*- coding: utf-8 -*-
"""

"""
from copy import deepcopy
from typing import Dict, Tuple

from numpy import zeros, ndarray, exp

from pyfem.fem.Timer import Timer
from pyfem.fem.constants import DTYPE
from pyfem.io.Material import Material
from pyfem.io.Section import Section
from pyfem.materials.BaseMaterial import BaseMaterial
from pyfem.materials.ElasticIsotropic import get_stiffness_from_young_poisson
from pyfem.utils.colors import error_style


class ViscoElasticMaxwell(BaseMaterial):

    def __init__(self, material: Material, dimension: int, section: Section) -> None:
        super().__init__(material, dimension, section)
        self.allowed_section_types = ('PlaneStress', 'PlaneStrain', 'Volume')
        self.E0: float = self.material.data[0]
        self.E1: float = self.material.data[1]
        self.E2: float = self.material.data[2]
        self.E3: float = self.material.data[3]
        self.TAU1: float = self.material.data[4]
        self.TAU2: float = self.material.data[5]
        self.TAU3: float = self.material.data[6]
        self.POISSON: float = self.material.data[7]
        self.create_tangent()

    def create_tangent(self):
        young = self.material.data[0]
        poisson = self.material.data[1]

        if self.section.type in self.allowed_section_types:
            self.ddsdde = get_stiffness_from_young_poisson(self.dimension, young, poisson, self.section.type)
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
            state_variable['SM1'] = zeros(ntens, dtype=DTYPE)
            state_variable['SM2'] = zeros(ntens, dtype=DTYPE)
            state_variable['SM3'] = zeros(ntens, dtype=DTYPE)

        SM1OLD = deepcopy(state_variable['SM1'])
        SM2OLD = deepcopy(state_variable['SM2'])
        SM3OLD = deepcopy(state_variable['SM3'])

        dtime = timer.dtime
        dstrain = variable['dstrain']
        strain = variable['strain']

        E0 = self.E0
        E1 = self.E1
        E2 = self.E2
        E3 = self.E3
        TAU1 = self.TAU1
        TAU2 = self.TAU2
        TAU3 = self.TAU3
        POISSON = self.POISSON

        mu0 = 0.5 * E0 / (1.0 + POISSON)
        mu1 = 0.5 * E1 / (1.0 + POISSON)
        mu2 = 0.5 * E2 / (1.0 + POISSON)
        mu3 = 0.5 * E3 / (1.0 + POISSON)

        BULK = E0 / 3 / (1.0 - 2 * POISSON)

        m1 = (TAU1 * mu1 - TAU1 * mu1 * exp(-dtime / TAU1)) / (mu0 * dtime)
        m2 = (TAU2 * mu2 - TAU2 * mu2 * exp(-dtime / TAU2)) / (mu0 * dtime)
        m3 = (TAU3 * mu3 - TAU3 * mu3 * exp(-dtime / TAU3)) / (mu0 * dtime)

        term1 = BULK + (4.0 * mu0) / 3.0
        term2 = BULK - (2.0 * mu0) / 3.0

        a1 = exp(-dtime / TAU1)
        a2 = exp(-dtime / TAU2)
        a3 = exp(-dtime / TAU3)

        g = zeros((ntens, ntens), dtype=DTYPE)

        for i in range(ndi):
            g[i, i] = term1
        for i in range(1, ndi):
            for j in range(0, i):
                g[i, j] = term2
                g[j, i] = term2
        for i in range(ndi, ntens):
            g[i, i] = mu0

        stress = zeros(ntens, dtype=DTYPE)

        for i in range(ndi):
            stress[i] = g[i, 0] * strain[0] + g[i, 1] * strain[1] + g[i, 2] * strain[2] + \
                        SM1OLD[i] + SM2OLD[i] + SM3OLD[i] + (1 + m1 + m2 + m3) * \
                        (g[i, 0] * dstrain[0] + g[i, 1] * dstrain[1] + g[i, 2] * dstrain[2])

        for i in range(ndi, ntens):
            stress[i] = g[i, i] * strain[i] + SM1OLD[i] + SM2OLD[i] + SM3OLD[i] + (1 + m1 + m2 + m3) * (
                    g[i, i] * dstrain[i])

        SM1 = zeros(ntens, dtype=DTYPE)
        SM2 = zeros(ntens, dtype=DTYPE)
        SM3 = zeros(ntens, dtype=DTYPE)

        for i in range(ndi):
            SM1[i] = SM1OLD[i] + m1 * g[i, 0] * dstrain[0] + g[i, 1] * dstrain[1] + g[i, 2] * dstrain[2]
            SM2[i] = SM2OLD[i] + m2 * g[i, 0] * dstrain[0] + g[i, 1] * dstrain[1] + g[i, 2] * dstrain[2]
            SM3[i] = SM3OLD[i] + m3 * g[i, 0] * dstrain[0] + g[i, 1] * dstrain[1] + g[i, 2] * dstrain[2]

        for i in range(ndi, ntens):
            SM1[i] = SM1OLD[i] + m1 * (g[i, i] * dstrain[i])
            SM2[i] = SM2OLD[i] + m2 * (g[i, i] * dstrain[i])
            SM3[i] = SM3OLD[i] + m3 * (g[i, i] * dstrain[i])

        for i in range(ntens):
            SM1[i] = exp(-dtime / TAU1) * SM1[i]
            SM2[i] = exp(-dtime / TAU2) * SM2[i]
            SM3[i] = exp(-dtime / TAU3) * SM3[i]

        state_variable_new['SM1'] = SM1
        state_variable_new['SM2'] = SM2
        state_variable_new['SM3'] = SM3

        ddsdde = (1 + m1 + m2 + m3) * g

        output = {'stress': stress}

        return ddsdde, output


if __name__ == "__main__":
    from pyfem.Job import Job

    job = Job(r'F:\Github\pyfem\examples\specimen3D\Job-1.toml')

    material_data = ViscoElasticMaxwell(job.props.materials[2], 3, job.props.sections[0])
    print(material_data.to_string())
