# -*- coding: utf-8 -*-
"""

"""
import os
import numpy as np

from pyfem.Job import Job
from pyfem.io.BaseIO import BaseIO
from pyfem.database.ODB import ODB

BaseIO.is_read_only = False

path = '../examples/mechanical/torch/'
# path = '../examples/mechanical/1element/quad4/'
path = '../examples/mechanical/plane/'

job = Job(os.path.join(path, 'Job-1.toml'))

# 参数设置
# job.props.show()

# 求解
job.assembly.__init__(job.props)
print(job.props.mesh_data.elements)
print(job.assembly.dof_solution)

stress = []
b = []
bt = []
dvol = []
element_dof_ids = []
element_dof_values = []
for element_data in job.assembly.element_data_list:
    stress.append(np.array(element_data.qp_stresses))
    b.append(element_data.qp_b_matrices)
    bt.append(element_data.qp_b_matrices_transpose)
    dvol.append(element_data.qp_weight_times_jacobi_dets)
    element_dof_ids.append(element_data.element_dof_ids)
    element_dof_values.append(job.assembly.dof_solution[element_data.element_dof_ids])

stress = np.array(stress)
b = np.array(b)
bt = np.array(bt)
dvol = np.array(dvol)
element_dof_values = np.array(element_dof_values)

print(b.shape, bt.shape, stress.shape, dvol.shape, element_dof_values.shape)

strain = b @ element_dof_values[:, np.newaxis, :, np.newaxis]
strain = np.einsum('...ik,...kj->...ij', b, element_dof_values[:, np.newaxis, :, np.newaxis])
print(strain.shape)

x = bt @ stress[:, :, :, np.newaxis]
x = np.einsum('...ik,...kj->...ij', bt, stress[:, :, :, np.newaxis])
print(x.shape)

y = np.squeeze(x, 3) * dvol[:, :, np.newaxis]
print(y.shape)

z = np.sum(y, axis=1)
print(z.shape)

fint = np.zeros(job.assembly.total_dof_number)
for i, conn in enumerate(element_dof_ids):
    fint[conn] += z[i]

# job.run_with_log()

# 后处理
odb = ODB()
odb.load_hdf5(os.path.join(path, 'Job-1.hdf5'))
for frame in odb.steps['Step-1']['frames']:
    print(type(frame['solution']['fint']))
