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
