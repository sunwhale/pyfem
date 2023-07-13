# -*- coding: utf-8 -*-
"""

"""
import pkgutil
import importlib
import inspect
import sys

PYFEM_PATH = r'F:\Github\pyfem\src'
sys.path.insert(0, PYFEM_PATH)

from pyfem.io.Amplitude import Amplitude

def list_modules(package_name):
    package = __import__(package_name)
    for importer, modname, ispkg in pkgutil.walk_packages(package.__path__, prefix=package.__name__ + '.'):
        print(importer, modname, ispkg)


def list_modules_and_variables(package_name):
    package = importlib.import_module(package_name)
    for importer, modname, ispkg in pkgutil.walk_packages(package.__path__, prefix=package.__name__ + '.'):
        module = importlib.import_module(modname)
        # print('Module:', modname)

        if 'materials.' in modname:
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and obj.__module__ == module.__name__:
                    print(modname, obj.__slots__)


            # for name, value in inspect.getmembers(module):
        #         print(modname, name, inspect.isclass(value))
        #
        #         if inspect.isclass(value):
        #             print(value.__slots__)

                # if not name.startswith('__'):
                #     print('  Variable:', name, '=', value)

list_modules_and_variables('pyfem')


# init_members = inspect.getmembers(Amplitude.__init__)
#
# print(Amplitude.__init__.__dir__())
#
# for name, _ in init_members:
#     print(name)




