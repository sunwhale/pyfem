from typing import Dict, List

from pyfem.io.Dofs import Dofs
from pyfem.io.Domain import Domain
from pyfem.io.Material import Material
from pyfem.io.Mesh import Mesh


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
        for key, item in self.__dict__.items():
            print(f'+-{key}')
            print(f'  |- {type(item)}')
            print(f'  |- {item}')
            if isinstance(item, list):
                for i, it in enumerate(item):
                    print(f'    |-{i}-{it}')
                    print(f'{it.to_string()}')

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


if __name__ == "__main__":
    props = Properties()
    props.show()
