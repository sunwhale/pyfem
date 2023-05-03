import time

from pyfem.io.arguments import get_arguments
from pyfem.io.parser import Properties


def main():
    inp_file_name, out_file_name, parameters = get_arguments()

    t1 = time.time()

    print(inp_file_name, out_file_name, parameters)

    props = Properties()
    props.read_toml(inp_file_name)
    props.print()

    t2 = time.time()

    total = t2 - t1
    print("Time elapsed = ", total, " [s].\n")

    print("Analysis terminated successfully.")
