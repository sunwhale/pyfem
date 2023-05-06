from typing import Dict, List

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from pyfem.io.Dofs import Dofs
from pyfem.io.Domain import Domain
from pyfem.io.Material import Material
from pyfem.io.Mesh import Mesh
from pyfem.io.BC import BC


class Properties:
    def __init__(self):
        self.toml = None
        self.title = None
        self.mesh = None
        self.dofs = None
        self.domains = None
        self.materials = None
        self.bcs = None
        self.solver = None
        self.output = None

    def show(self) -> None:
        CYAN = '\033[36m'
        MAGENTA = '\033[35m'
        BLUE = '\033[34m'
        GREEN = '\033[32m'
        YELLOW = '\033[33m'
        RED = '\033[31m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'
        END = '\033[0m'

        for key, item in self.__dict__.items():
            print()
            print(CYAN + f'+-{key}' + END)
            print(MAGENTA + f'  |- {type(item)}' + END)
            try:
                print(f'  |- {item.to_string()}')
            except:
                print(f'  |- {item}')
            if isinstance(item, list):
                for i, it in enumerate(item):
                    # print(BLUE + f'    |-{i}-{it}' + END)
                    print(f'    |-{i}-{it.to_string()}')

    def set_toml(self, toml: Dict) -> None:
        self.toml = toml

    def set_title(self, title: str) -> None:
        self.title = title

    def set_mesh(self, mesh_dict: Dict) -> None:
        self.mesh = Mesh()
        allowed_keys = self.mesh.__dict__.keys()
        for key, item in mesh_dict.items():
            if key in allowed_keys:
                self.mesh.__setattr__(key, item)
            else:
                raise KeyError(f'{key} is not the keyword of mesh.')

    def set_dofs(self, dofs_dict: Dict) -> None:
        self.dofs = Dofs()
        allowed_keys = self.dofs.__dict__.keys()
        for key, item in dofs_dict.items():
            if key in allowed_keys:
                self.dofs.__setattr__(key, item)
            else:
                raise KeyError(f'{key} is not the keyword of dofs.')

    def set_materials(self, materials_list: List) -> None:
        self.materials = []
        for material_dict in materials_list:
            material = Material()
            allowed_keys = material.__dict__.keys()
            for key, item in material_dict.items():
                if key in allowed_keys:
                    material.__setattr__(key, item)
                else:
                    raise KeyError(f'{key} is not the keyword of materials.')
            self.materials.append(material)

    def set_domains(self, domains_list: List) -> None:
        self.domains = []
        for domain_dict in domains_list:
            domain = Domain()
            allowed_keys = domain.__dict__.keys()
            for key, item in domain_dict.items():
                if key in allowed_keys:
                    domain.__setattr__(key, item)
                else:
                    raise KeyError(f'{key} is not the keyword of domains.')
            self.domains.append(domain)

    def set_bcs(self, bcs_list: List) -> None:
        self.bcs = []
        for bc_dict in bcs_list:
            bc = BC()
            allowed_keys = bc.__dict__.keys()
            for key, item in bc_dict.items():
                if key in allowed_keys:
                    bc.__setattr__(key, item)
                else:
                    raise KeyError(f'{key} is not the keyword of bcs.')
            self.bcs.append(bc)

    def read_file(self, file_name: str) -> None:
        with open(file_name, "rb") as f:
            toml = tomllib.load(f)
            self.set_toml(toml)

        toml_keys = self.toml.keys()
        allowed_keys = self.__dict__.keys()

        for key in toml_keys:
            if key not in allowed_keys:
                raise KeyError(f'{key} is not the keyword of properties.')

        if 'title' in toml_keys:
            title = self.toml['title']
            self.set_title(title)

        if 'mesh' in toml_keys:
            mesh_dict = self.toml['mesh']
            self.set_mesh(mesh_dict)

        if 'dofs' in toml_keys:
            dofs_dict = self.toml['dofs']
            self.set_dofs(dofs_dict)

        if 'domains' in toml_keys:
            domains_list = self.toml['domains']
            self.set_domains(domains_list)

        if 'materials' in toml_keys:
            materials_list = self.toml['materials']
            self.set_materials(materials_list)

        if 'bcs' in toml_keys:
            bcs_list = self.toml['bcs']
            self.set_bcs(bcs_list)


if __name__ == "__main__":
    props = Properties()
    # props.show()
    props.read_file(r'F:\Github\pyfem\examples\rectangle\rectangle.toml')
    props.show()
    # bc = props.bcs[0]
    # print(bc.to_string())
