import time

from pyfem.io.Properties import Properties
from pyfem.io.arguments import get_arguments


def main():
    inp_file_name, out_file_name, parameters = get_arguments()

    t1 = time.time()

    print(inp_file_name, out_file_name, parameters)

    props = Properties()
    props.read_file(inp_file_name)
    props.show()

    t2 = time.time()

    total = t2 - t1
    print("Time elapsed = ", total, " [s].\n")

    print("Analysis terminated successfully.")
