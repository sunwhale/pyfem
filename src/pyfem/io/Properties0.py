from typing import Dict, List, Optional

try:
    import tomllib  # type: ignore
except ModuleNotFoundError:
    import tomli as tomllib

from pyfem.io.Dofs import Dofs
from pyfem.io.Domain import Domain
from pyfem.io.Material import Material
from pyfem.io.Mesh import Mesh
from pyfem.io.BC import BC
from pyfem.io.Solver import Solver
from pyfem.io.Output import Output


class Properties:
    __slots__ = ('__toml', '__title', '__mesh', '__dofs', '__domains', '__materials', '__bcs', '__solver', '__outputs')

    def __init__(self):
        self.__toml = None
        self.__title = None
        self.__mesh = None
        self.__dofs = None
        self.__domains = None
        self.__materials = None
        self.__bcs = None
        self.__solver = None
        self.__outputs = None

    @property
    def toml(self) -> Optional[Dict]:
        return self.__toml

    @property
    def title(self) -> Optional[str]:
        return self.__title

    @property
    def mesh(self) -> Optional[Mesh]:
        return self.__mesh

    @property
    def dofs(self) -> Optional[Dofs]:
        return self.__dofs

    @property
    def domains(self) -> Optional[List[Domain]]:
        return self.__domains

    @property
    def materials(self) -> Optional[List[Material]]:
        return self.__materials

    @property
    def bcs(self) -> Optional[List[BC]]:
        return self.__bcs

    @property
    def solver(self) -> Optional[Solver]:
        return self.__solver

    @property
    def outputs(self) -> Optional[List[Output]]:
        return self.__outputs

    def show(self) -> None:
        CYAN = '\033[36m'
        MAGENTA = '\033[35m'
        BLUE = '\033[34m'
        END = '\033[0m'
        for key in self.__slots__:
            item = self.__getattribute__(f'_Properties{key}')
            print()
            print(CYAN + f'+-{key[2:]}' + END)
            print(MAGENTA + f'  |- {type(item)}' + END)
            if hasattr(item, 'to_string'):
                print(f'  |- {item.to_string()}')
            else:
                print(f'  |- {item}')
            if isinstance(item, list):
                for i, it in enumerate(item):
                    print(BLUE + f'    |-{i}-{it.to_string(level=3)}' + END)

    def set_toml(self, toml: Dict) -> None:
        self.__toml = toml

    def set_title(self, title: str) -> None:
        self.__title = title

    def set_mesh(self, mesh_dict: Dict) -> None:
        self.__mesh = Mesh()
        allowed_keys = self.__mesh.__dict__.keys()
        for key, item in mesh_dict.items():
            if key in allowed_keys:
                self.__mesh.__setattr__(key, item)
            else:
                raise KeyError(f'{key} is not the keyword of mesh.')

    def set_dofs(self, dofs_dict: Dict) -> None:
        self.__dofs = Dofs()
        allowed_keys = self.__dofs.__dict__.keys()
        for key, item in dofs_dict.items():
            if key in allowed_keys:
                self.__dofs.__setattr__(key, item)
            else:
                raise KeyError(f'{key} is not the keyword of dofs.')

    def set_solver(self, solver_dict: Dict) -> None:
        self.__solver = Solver()
        allowed_keys = self.__solver.__dict__.keys()
        for key, item in solver_dict.items():
            if key in allowed_keys:
                self.__solver.__setattr__(key, item)
            else:
                raise KeyError(f'{key} is not the keyword of solver.')

    def set_materials(self, materials_list: List) -> None:
        self.__materials = []
        for material_dict in materials_list:
            material = Material()
            allowed_keys = material.__dict__.keys()
            for key, item in material_dict.items():
                if key in allowed_keys:
                    material.__setattr__(key, item)
                else:
                    raise KeyError(f'{key} is not the keyword of materials.')
            self.__materials.append(material)

    def set_domains(self, domains_list: List) -> None:
        self.__domains = []
        for domain_dict in domains_list:
            domain = Domain()
            allowed_keys = domain.__dict__.keys()
            for key, item in domain_dict.items():
                if key in allowed_keys:
                    domain.__setattr__(key, item)
                else:
                    raise KeyError(f'{key} is not the keyword of domains.')
            self.__domains.append(domain)

    def set_bcs(self, bcs_list: List) -> None:
        self.__bcs = []
        for bc_dict in bcs_list:
            bc = BC()
            allowed_keys = bc.__dict__.keys()
            for key, item in bc_dict.items():
                if key in allowed_keys:
                    bc.__setattr__(key, item)
                else:
                    raise KeyError(f'{key} is not the keyword of bcs.')
            self.__bcs.append(bc)

    def set_outputs(self, outputs_list: List) -> None:
        self.__outputs = []
        for output_dict in outputs_list:
            output = Output()
            allowed_keys = output.__dict__.keys()
            for key, item in output_dict.items():
                if key in allowed_keys:
                    output.__setattr__(key, item)
                else:
                    raise KeyError(f'{key} is not the keyword of bcs.')
            self.__outputs.append(output)

    def read_file(self, file_name: str) -> None:
        with open(file_name, "rb") as f:
            toml = tomllib.load(f)
            self.set_toml(toml)

        toml_keys = self.__toml.keys()
        allowed_keys = self.__slots__

        for key in toml_keys:
            if f'__{key}' not in allowed_keys:
                raise KeyError(f'{key} is not the keyword of properties.')

        if 'title' in toml_keys:
            title = self.__toml['title']
            self.set_title(title)

        if 'mesh' in toml_keys:
            mesh_dict = self.__toml['mesh']
            self.set_mesh(mesh_dict)

        if 'dofs' in toml_keys:
            dofs_dict = self.__toml['dofs']
            self.set_dofs(dofs_dict)

        if 'domains' in toml_keys:
            domains_list = self.__toml['domains']
            self.set_domains(domains_list)

        if 'materials' in toml_keys:
            materials_list = self.__toml['materials']
            self.set_materials(materials_list)

        if 'bcs' in toml_keys:
            bcs_list = self.__toml['bcs']
            self.set_bcs(bcs_list)

        if 'solver' in toml_keys:
            solver_dict = self.__toml['solver']
            self.set_solver(solver_dict)

        if 'outputs' in toml_keys:
            outputs_dict = self.__toml['outputs']
            self.set_outputs(outputs_dict)


if __name__ == "__main__":
    props = Properties()
    props.show()
    props.read_file(r'F:\Github\pyfem\examples\rectangle\rectangle.toml')
    # material = props.materials[0]
    # print(material.to_string())
    #
    # props.show()

    props.title = 1
