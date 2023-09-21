# -*- coding: utf-8 -*-
"""

"""
import sys

PYFEM_PATH = r'/'
sys.path.insert(0, PYFEM_PATH)

import os

from pyfem.Job import Job


def get_files_with_name(directory, file_name):
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == file_name:
                file_path = os.path.join(root, file)
                file_paths.append(file_path)
    return file_paths


# 指定目录路径和要查找的文件名
directory_path = r"../examples/mechanical_phase"
file_name = "Job-1.toml"

# 调用函数来获取文件路径
file_paths = get_files_with_name(directory_path, file_name)
# file_paths = [r'../examples\mechanical_phase\rectangle\Job-1.toml']

# 打印文件路径
for file_path in file_paths:
    print(file_path)
    job = Job(file_path)
    if job.run() == 0:
        print('Pass')
    else:
        print('Error')
