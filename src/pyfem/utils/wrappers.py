import inspect
import time

from pyfem.utils.colors import BOLD, MAGENTA, YELLOW, BLUE, END


def show_running_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        running_time = end_time - start_time
        print(BOLD + MAGENTA + f'{func} running time ' + END + YELLOW + f'= {running_time} s.' + END)
        return result

    return wrapper


def trace_calls(func):
    def wrapper(*args, **kwargs):
        call_stack = inspect.stack()
        call_frames = []
        for frame in call_stack[1:]:
            filename = frame.filename.split('src')[-1]
            call_frames.append((filename, frame.function, frame.lineno))
        print(f'{BLUE}{func} called from:{END}')
        for frame in call_frames:
            print(f'{YELLOW}\t{frame[0]}:{END}{frame[1]}: line {frame[2]}')
        return func(*args, **kwargs)

    return wrapper
