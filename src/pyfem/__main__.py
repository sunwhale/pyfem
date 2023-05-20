from pyfem.assembly.Assembly import Assembly
from pyfem.io.Properties import Properties
from pyfem.io.arguments import get_arguments
from pyfem.io.write_vtu import write_vtk
from pyfem.solvers.get_solver_data import get_solver_data
from pyfem.utils.wrappers import show_running_time


@show_running_time
def main():
    args = get_arguments()

    props = Properties()

    props.read_file(args.i)

    props.verify()

    # props.show()

    assembly = Assembly(props)

    solver_data = get_solver_data(assembly, props.solver)

    solver_data.run()

    write_vtk(props, assembly)

    print("Analysis terminated successfully.")
