# -*- coding: utf-8 -*-
"""

"""
import os
import shutil


def copy_py_files(src_dir, dest_dir):
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.py'):
                src_file = os.path.join(root, file)
                dest_subdir = os.path.relpath(root, src_dir)
                dest_file = os.path.join(dest_dir, dest_subdir, file)

                dest_subdir_path = os.path.join(dest_dir, dest_subdir)
                if not os.path.exists(dest_subdir_path):
                    os.makedirs(dest_subdir_path)

                shutil.copy(src_file, dest_file)
                print(f"Copying {src_file} to {dest_file}")


# 源目录A
src_directory = r'F:\Github\pyfem\src\pyfem'
# 目标目录B
dest_directory = r'C:\Users\SunJingyu\.conda\envs\gui311\Lib\site-packages\pyfem'

copy_py_files(src_directory, dest_directory)

