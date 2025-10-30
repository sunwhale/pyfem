import os
import pathlib

import petsc4py


def get_petsc_lib() -> pathlib.Path:
    """Find the full path of the PETSc shared library.

    Returns:
        Full path to the PETSc shared library.

    Raises:
        RuntimeError: If PETSc library cannot be found for if more than
            one library is found.
    """
    import petsc4py as _petsc4py

    petsc_dir = _petsc4py.get_config()["PETSC_DIR"]
    petsc_arch = _petsc4py.lib.getPathArchPETSc()[1]
    candidate_paths = [
        os.path.join(petsc_dir, petsc_arch, "lib", "libpetsc.so"),
        os.path.join(petsc_dir, petsc_arch, "lib", "libpetsc.dylib"),
    ]
    exists_paths = []
    for candidate_path in candidate_paths:
        if os.path.exists(candidate_path):
            exists_paths.append(candidate_path)

    if len(exists_paths) == 0:
        raise RuntimeError("Could not find a PETSc shared library.")
    elif len(exists_paths) > 1:
        raise RuntimeError("More than one PETSc shared library found.")

    return pathlib.Path(exists_paths[0])


if __name__ == "__main__":
    print(petsc4py.get_config())
    print(petsc4py.get_include())