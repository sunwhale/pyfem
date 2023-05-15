import time

from pyfem.io.Properties import Properties
from pyfem.io.arguments import get_arguments
from pyfem.utils.wrappers import show_running_time


@show_running_time
def main():
    inp_file_name, out_file_name, parameters = get_arguments()

    props = Properties()
    props.read_file(inp_file_name)
    # props.show()

    print("Analysis terminated successfully.")
