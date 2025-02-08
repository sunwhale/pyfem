# -*- coding: utf-8 -*-
"""

"""
import os
from pyfem.Job import Job
from pyfem.io.BaseIO import BaseIO
from pyfem.database.ODB import ODB
import matplotlib.pyplot as plt

BaseIO.is_read_only = False

path = ''

job = Job(os.path.join(path, 'Job-1.toml'))

# 参数设置
job.props.solver.total_time = 1.0
job.props.bcs[3].value = -250.0
job.props.materials[0].data_dict['p_s'] = [10.0]
job.props.show()

# 求解
job.assembly.__init__(job.props)
job.run_with_log()

# 后处理
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