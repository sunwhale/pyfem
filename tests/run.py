# -*- coding: utf-8 -*-
"""

"""
import sys
import importlib

PYFEM_PATH = r'F:\Github\pyfem\src'
sys.path.insert(0, PYFEM_PATH)

from pyfem.Job import Job
from pyfem.io.BaseIO import BaseIO

job = Job(r'..\examples\mechanical_phase\1element\hex8_visco\Job-1.toml')

BaseIO.is_read_only = False
job.props.materials[1].data = [0.001, 0.001]
job.assembly.__init__(job.props)

_, x, y = job.run()

print(x)
print(y)

# umat = importlib.import_module('UMAT')

# print(umat.UMAT)
