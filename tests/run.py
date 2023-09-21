# -*- coding: utf-8 -*-
"""

"""
import sys
import matplotlib.pyplot as plt

PYFEM_PATH = r'F:\Github\pyfem\src'
sys.path.insert(0, PYFEM_PATH)

from pyfem.Job import Job
from pyfem.io.BaseIO import BaseIO

# job = Job(r'..\examples\mechanical_phase\1element\hex8_visco\Job-1.toml')
job = Job(r'..\examples\mechanical\1element\hex8_visco\Job-1.toml')

BaseIO.is_read_only = False
job.props.materials[0].data = [0.96, 0.14, 2.61e3, 6.57e-5, 5.35, 0.639, 1.15, 2000.0]
job.props.materials[1].data = [0.009, 0.001]
job.props.solver.total_time = 2000.0
job.props.solver.max_dtime = job.props.solver.total_time / 50.0
job.props.solver.initial_dtime = job.props.solver.total_time / 50.0
job.props.amplitudes[0].data = [
    [0.0, 0.0],
    [job.props.solver.total_time, 1.0]
]
job.assembly.__init__(job.props)

_, x, y = job.run()

plt.plot(x, y)
plt.show()
