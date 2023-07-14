# -*- coding: utf-8 -*-
"""

"""
import pkgutil
import importlib
import inspect
import sys

PYFEM_PATH = r'F:\Github\pyfem\src'
sys.path.insert(0, PYFEM_PATH)


def list_modules_and_variables(package_name):
    package = importlib.import_module(package_name)
    for importer, module_name, is_package in pkgutil.walk_packages(package.__path__, prefix=package.__name__ + '.'):
        module = importlib.import_module(module_name)

        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and obj.__module__ == module.__name__:
                print(module_name)
                indent = 1
                for attribute in obj.__slots__:
                    print(' ' * (indent + 4) + attribute)



list_modules_and_variables('pyfem')

import graphviz

# def get_package_tree(package_name):
#     # 创建一个有向图
#     graph = graphviz.Digraph()
#
#     # 递归遍历包内的模块和子包
#     def traverse_package(package_name, parent_node=None):
#         # 获取包内的模块和子包
#         package = __import__(package_name, fromlist=["dummy"])
#         modules = pkgutil.walk_packages(package.__path__, package.__name__ + ".")
#
#         # 遍历模块和子包
#         for loader, module_name, is_pkg in modules:
#             # 添加节点
#             if is_pkg:
#                 # 子包节点
#                 node_label = module_name.split(".")[-1]
#                 node_name = module_name.replace(".", "_")
#                 graph.node(node_name, label=node_label, shape="box")
#             else:
#                 # 模块节点
#                 node_label = module_name.split(".")[-1]
#                 node_name = module_name.replace(".", "_")
#                 graph.node(node_name, label=node_label, shape="ellipse")
#
#             # 添加边
#             if parent_node is not None:
#                 graph.edge(parent_node, node_name)
#
#             # 递归遍历子包
#             if is_pkg:
#                 traverse_package(module_name, node_name)
#
#     # 开始遍历
#     traverse_package(package_name)
#
#     # 渲染图形
#     graph.render(view=True)

# 示例用法
# get_package_tree("pyfem")

# import importlib
# import inspect
#
#
# def get_module_tree(package_name, indent=0):
#     package = importlib.import_module(package_name)
#     print(' ' * indent + package_name)
#
#     for _, module_name, is_pkg in pkgutil.iter_modules(package.__path__):
#         if is_pkg:
#             get_module_tree(package_name + '.' + module_name, indent + 4)
#         else:
#             module = importlib.import_module(package_name + '.' + module_name)
#             print(' ' * (indent + 4) + module_name)
#
#             for _, obj in inspect.getmembers(module):
#                 if inspect.isfunction(obj):
#                     print(' ' * (indent + 8) + obj.__name__)
#
#
# get_module_tree('pyfem')

