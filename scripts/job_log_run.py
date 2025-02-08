# -*- coding: utf-8 -*-
"""

"""
import os
from pyfem.Job import Job
from pyfem.io.BaseIO import BaseIO
from pyfem.database.ODB import ODB
import matplotlib.pyplot as plt

BaseIO.is_read_only = False

path = r'F:\Github\pyfem\examples\mechanical\rectangle_hole_3D'

job = Job(os.path.join(path, 'Job-1.toml'))
job.props.solver.total_time = 1.0
job.props.materials[0].data_dict['p_s'] = 20.0
job.props.materials[0].show()
job.log_run()

odb = ODB()
odb.load_hdf5(os.path.join(path, 'Job-1.hdf5'))

t = []
e11 = []
s11 = []
u11 = []
for frame in odb.steps['Step-1']['frames']:
    t.append(frame['frameValue'])
    e11.append(frame['fieldOutputs']['E11']['bulkDataBlocks'][20])
    s11.append(frame['fieldOutputs']['S11']['bulkDataBlocks'][20])
    u11.append(frame['fieldOutputs']['U']['bulkDataBlocks'][20][0])

plt.plot(e11, s11, label='E11')
plt.show()