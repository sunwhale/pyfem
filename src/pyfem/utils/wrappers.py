# -*- coding: utf-8 -*-
"""
定义一些函数装饰器
"""
import inspect
import time

from pyfem.utils.colors import BOLD, MAGENTA, YELLOW, BLUE, END
from pyfem.utils.logger import logger


def show_running_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        running_time = end_time - start_time
        # logger.debug(BOLD + MAGENTA + f'{func} running time ' + END + YELLOW + f'= {running_time} s.' + END)
        logger.log(22, f'Function {func} running time {running_time} s.')
        return result

    return wrapper


def trace_calls(func):
    def wrapper(*args, **kwargs):
        call_stack = inspect.stack()
        call_frames = []
        for frame in call_stack[1:]:
            filename = frame.filename.split('src')[-1]
            call_frames.append((filename, frame.function, frame.lineno))
        logger.debug(f'{BLUE}{func} called from:{END}')
        for frame in call_frames:
            logger.debug(f'{YELLOW}\t{frame[0]}:{END}{frame[1]}: line {frame[2]}')
        return func(*args, **kwargs)

    return wrapper
