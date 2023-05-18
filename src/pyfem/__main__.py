from pyfem.io.Properties import Properties
from pyfem.io.arguments import get_arguments
from pyfem.assembly.Assembly import Assembly
from pyfem.utils.wrappers import show_running_time


@show_running_time
def main():
    inp_file_name, out_file_name, parameters = get_arguments()

    props = Properties()
    props.read_file(inp_file_name)
    props.verify()
    props.show()

    assembly = Assembly(props)

    print(assembly.global_stiffness.shape)

    print("Analysis terminated successfully.")
