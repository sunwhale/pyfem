import time

from pyfem.utils.colors import BOLD, MAGENTA, YELLOW, END


def show_running_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        running_time = end_time - start_time
        print(BOLD + MAGENTA + f'{func} running time ' + END + YELLOW + f'= {running_time} s.' + END)
        return result

    return wrapper
