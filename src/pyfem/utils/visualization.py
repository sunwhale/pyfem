# -*- coding: utf-8 -*-
"""
定义一些函数用于程序的可视化。
"""
from numpy import ndarray

from pyfem.utils.colors import GREEN, BLUE, END


def insert_spaces(n: int, text: str) -> str:
    lines = text.split('\n')
    indented_lines = [' ' * n + line for line in lines]
    return '\n'.join(indented_lines)


def object_dict_to_string(obj, level: int = 1) -> str:
    msg = BLUE + obj.__str__() + END
    msg += '\n'
    for key, item in obj.__dict__.items():
        msg += '  ' * level + f'|- {key}: {item}\n'
    return msg[:-1]


def object_dict_to_string_ndarray(obj, level: int = 1) -> str:
    msg = BLUE + obj.__str__() + END
    msg += '\n'
    for key, item in obj.__dict__.items():
        if isinstance(item, ndarray):
            msg += '  ' * level + GREEN + f'|- {key}: ' + END + f'{type(item)} with shape = {item.shape} \n'
            msg += insert_spaces(5 + (level - 1) * 2, f'{item}') + '\n'
        else:
            msg += '  ' * level + GREEN + f'|- {key}: ' + END + f'{item}\n'
    return msg[:-1]


def object_dict_to_string_assembly(obj, level: int = 1) -> str:
    msg = BLUE + obj.__str__() + END
    msg += '\n'
    for key, item in obj.__dict__.items():
        if isinstance(item, list) and len(item) > 8:
            msg += '  ' * level + GREEN + f'|- {key}: ' + END + f'{type(item)} of with length = {len(item)} \n'
        elif isinstance(item, ndarray):
            msg += '  ' * level + GREEN + f'|- {key}: ' + END + f'{type(item)} with shape = {item.shape} \n'
        elif key == 'global_stiffness':
            msg += '  ' * level + GREEN + f'|- {key}: ' + END + f'{type(item)} with shape = {item.shape} \n'
        else:
            msg += '  ' * level + GREEN + f'|- {key}: ' + END + f'{item}\n'
    return msg[:-1]


def object_slots_to_string(obj, level: int = 1) -> str:
    msg = BLUE + obj.__str__() + END
    msg += '\n'
    for key in obj.__slots__:
        item = obj.__getattribute__(key)
        msg += '  ' * level + f'|- {key}: {item}\n'
    return msg[:-1]


def object_slots_to_string_ndarray(obj, level: int = 1) -> str:
    msg = BLUE + obj.__str__() + END
    msg += '\n'
    for key in obj.__slots__:
        item = obj.__getattribute__(key)
        if isinstance(item, ndarray):
            msg += '  ' * level + GREEN + f'|- {key}: ' + END + f'{type(item)} with shape = {item.shape} \n'
            msg += insert_spaces(5 + (level - 1) * 2, f'{item}') + '\n'
        else:
            msg += '  ' * level + GREEN + f'|- {key}: ' + END + f'{item}\n'
    return msg[:-1]


def object_slots_to_string_assembly(obj, level: int = 1) -> str:
    msg = BLUE + obj.__str__() + END
    msg += '\n'
    for key in obj.__slots__:
        item = obj.__getattribute__(key)
        if isinstance(item, list) and len(item) > 8:
            msg += '  ' * level + GREEN + f'|- {key}: ' + END + f'{type(item)} of with length = {len(item)} \n'
        elif isinstance(item, ndarray):
            msg += '  ' * level + GREEN + f'|- {key}: ' + END + f'{type(item)} with shape = {item.shape} \n'
        elif key == 'global_stiffness':
            msg += '  ' * level + GREEN + f'|- {key}: ' + END + f'{type(item)} with shape = {item.shape} \n'
        else:
            msg += '  ' * level + GREEN + f'|- {key}: ' + END + f'{item}\n'
    return msg[:-1]


def get_ordinal_number(num: int) -> str:
    if num % 100 in [11, 12, 13]:
        return str(num) + "th"
    elif num % 10 == 1:
        return str(num) + "st"
    elif num % 10 == 2:
        return str(num) + "nd"
    elif num % 10 == 3:
        return str(num) + "rd"
    else:
        return str(num) + "th"


def print_slots_dict(slots_dict: dict) -> None:
    for key, item in slots_dict.items():
        print(f'    :ivar {key}: {item[1]}')
        print(f'    :vartype {key}: {item[0]}\n')
