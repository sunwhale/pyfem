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
        if isinstance(item, list):
            msg += '  ' * level + GREEN + f'|- {key}: ' + END + f'{type(item)} with length = {len(item)} \n'
        elif isinstance(item, ndarray):
            msg += '  ' * level + GREEN + f'|- {key}: ' + END + f'{type(item)} with shape = {item.shape} \n'
        elif key == 'global_stiffness':
            msg += '  ' * level + GREEN + f'|- {key}: ' + END + f'{type(item)} with shape = {item.shape} \n'
        else:
            msg += '  ' * level + GREEN + f'|- {key}: ' + END + f'{item}\n'
    return msg[:-1]