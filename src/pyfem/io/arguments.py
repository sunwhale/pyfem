# -*- coding: utf-8 -*-
"""

"""
import sys
from argparse import ArgumentParser, Namespace, SUPPRESS

from pyfem import __version__


def get_arguments() -> Namespace:
    # 创建一个 argparse 解析器对象
    parser = ArgumentParser(add_help=False)

    # 添加程序输入文件选项
    parser.add_argument('-i', metavar='input', type=str,
                        help='Identify the input file.')

    # 添加程序输入文件选项
    parser.add_argument('-u', metavar='user', type=str,
                        help='User defined module.')

    # 添加程序输出文件选项
    parser.add_argument('-o', metavar='output', type=str,
                        help='Identify the output file.')

    # 添加参数选项
    parser.add_argument('-p', metavar='parameter', type=str,
                        help='Parameter to pass to the program.')

    # 添加帮助选项
    parser.add_argument('-h', '--help', action='help', default=SUPPRESS,
                        help='Show this help message and exit.')

    # 添加版本选项
    parser.add_argument('-v', '--version', action='version', help='Show program\'s version number and exit.',
                        version=f'pyfem {__version__}')

    # 解析命令行参数
    args = parser.parse_args()

    # 如果未指定程序输入文件，则打印帮助并退出
    if not args.i:
        print('--------------------------------------')
        print('>>> error: the input file is required.')
        print('--------------------------------------')
        parser.print_help()
        sys.exit()

    return args
