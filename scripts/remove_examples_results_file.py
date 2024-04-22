# -*- coding: utf-8 -*-
"""

"""
import os


def delete_files_with_extensions(directory, extensions):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.endswith(extension) for extension in extensions):
                file_path = os.path.join(root, file)
                os.remove(file_path)
                print(f"Deleted file: {file_path}")


# 指定目录路径和要删除的文件扩展名
directory_path = "../tests/validated_examples"
file_extensions = [".pvd", ".vtu", ".rpy", ".rec", ".jnl", ".sta", ".log", ".sta"]
# 调用函数来删除文件
delete_files_with_extensions(directory_path, file_extensions)

# 指定目录路径和要删除的文件扩展名
directory_path = "../examples"
file_extensions = [".pvd", ".vtu", ".rpy", ".rec", ".jnl", ".sta", ".log", ".sta"]
# 调用函数来删除文件
delete_files_with_extensions(directory_path, file_extensions)
