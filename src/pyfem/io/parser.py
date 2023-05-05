try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from pyfem.io.Properties import Properties


def read_toml(props: Properties, file_name: str) -> None:
    with open(file_name, "rb") as f:
        toml = tomllib.load(f)
        props.set_toml(toml)

    toml_keys = props.toml.keys()
    allowed_keys = props.__dict__.keys()

    for key in toml_keys:
        if key not in allowed_keys:
            raise KeyError(f'{key} is not the keyword of properties.')

    if 'title' in toml_keys:
        title = props.toml['title']
        props.set_title(title)

    if 'mesh' in toml_keys:
        mesh_dict = props.toml['mesh']
        props.set_mesh(mesh_dict)

    if 'dofs' in toml_keys:
        dofs_dict = props.toml['dofs']
        props.set_dofs(dofs_dict)

    if 'domains' in toml_keys:
        domains_list = props.toml['domains']
        props.set_domains(domains_list)

    if 'materials' in toml_keys:
        materials_list = props.toml['materials']
        props.set_materials(materials_list)

    if 'bcs' in toml_keys:
        bcs_list = props.toml['bcs']
        props.set_bcs(bcs_list)



if __name__ == "__main__":
    props = Properties()
    read_toml(props, r'F:\Github\pyfem\examples\rectangle\rectangle.toml')
    # props.show()
    print(props.materials[0].type)
