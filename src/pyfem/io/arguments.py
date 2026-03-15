# -*- coding: utf-8 -*-
"""

"""
import sys
from argparse import ArgumentParser, Namespace, SUPPRESS

from pyfem import __version__

# 全局缓存，用于存储解析后的命令行参数
_args = None

# 创建一个 argparse 解析器对象
_parser = ArgumentParser(add_help=False)


def parse_arguments() -> Namespace:
    """
    解析命令行参数，返回 Namespace 对象，并缓存结果。
    """
    global _args
    if _args is not None:
        return _args

    # 添加程序输入文件选项
    _parser.add_argument('-i', metavar='input', type=str,
                        help='Identify the input file.')

    # 添加程序输入文件选项
    _parser.add_argument('-u', metavar='user', type=str,
                        help='User defined module.')

    # 添加程序输出文件选项
    _parser.add_argument('-o', metavar='output', type=str,
                        help='Identify the output file.')

    # 添加参数选项
    _parser.add_argument('-p', metavar='parameter', type=str,
                        help='Parameter to pass to the program.')

    # 添加帮助选项
    _parser.add_argument('-h', '--help', action='help', default=SUPPRESS,
                        help='Show this help message and exit.')

    # 添加版本选项
    _parser.add_argument('-v', '--version', action='version', help='Show program\'s version number and exit.',
                        version=f'pyfem {__version__}')

    _parser.add_argument('--petsc', action='store_true', help='Enable PETSc support.')

    _parser.add_argument('--mpi', action='store_true', help='Enable MPI support.')

    _parser.add_argument('--debug', action='store_true', help='Enable DEBUG color support.')

    # 解析命令行参数
    args = _parser.parse_args()

    _args = args
    return _args


def get_arguments() -> Namespace:
    """
    获取解析后的命令行参数。
    如果尚未解析，则自动调用 parse_args() 进行解析。
    """
    if _args is None:
        return parse_arguments()
    return _args


def print_usage_and_exit():
    """打印错误信息和帮助，并退出程序（适用于命令行工具）。"""
    print('--------------------------------------')
    print('>>> error: the input file is required.')
    print('--------------------------------------')
    _parser.print_help()
    sys.exit(1)
