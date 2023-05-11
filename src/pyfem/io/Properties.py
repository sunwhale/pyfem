from typing import Dict, List, Any

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
from pyfem.fem.NodeSet import NodeSet
from pyfem.fem.ElementSet import ElementSet
from pyfem.utils.colors import CYAN, MAGENTA, BLUE, END, BOLD, error_style


class Properties:
    """
    Properties类用于解析配置文件中定义的属性。
    当 self.is_read_only = True 时：

    1. Properties 类的所有属性在首次被赋非None值后不能再被修改和删除，

    2. 此时许可的属性关键字存储在self.slots中。
    """
    is_read_only = True
    slots = ('toml', 'title', 'mesh', 'dofs', 'domains', 'materials', 'bcs', 'solver', 'outputs', 'nodes', 'elements')

    def __init__(self):
        self.toml = None
        self.title = None
        self.mesh = None
        self.dofs = None
        self.domains = None
        self.materials = None
        self.bcs = None
        self.solver = None
        self.outputs = None
        self.nodes = None
        self.elements = None

    def __setattr__(self, key, value):
        if self.is_read_only:
            if key not in self.slots:
                error_msg = f'{key} is not an allowable attribute keyword of {type(self).__name__}'
                raise AttributeError(error_style(error_msg))
            elif hasattr(self, key) and self.__getattribute__(key) is not None:
                error_msg = f'attribute {type(self).__name__}.{key} is READ ONLY'
                raise PermissionError(error_style(error_msg))
            else:
                super().__setattr__(key, value)
        else:
            super().__setattr__(key, value)

    def __delattr__(self, key):
        if self.is_read_only:
            error_msg = f'attribute {type(self).__name__}.{key} is READ ONLY'
            raise PermissionError(error_style(error_msg))
        else:
            super().__delattr__(key)

    def show(self) -> None:
        for key, item in self.__dict__.items():
            print()
            print(CYAN + f'+-{key}' + END)
            print(MAGENTA + f'  |- {type(item)}' + END)
            if hasattr(item, 'to_string'):
                print(f'  |- {item.to_string()}')
            else:
                print(f'  |- {item}')
            if isinstance(item, list):
                for i, it in enumerate(item):
                    print(BLUE + f'    |-{i}-{it.to_string(level=3)}' + END)

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
                raise AttributeError(self.key_error_message(key, self.mesh))
        if self.mesh.type == 'gmsh':
            self.set_nodes_from_gmsh()
            self.set_elements_from_gmsh()

    def set_dofs(self, dofs_dict: Dict) -> None:
        self.dofs = Dofs()
        allowed_keys = self.dofs.__dict__.keys()
        for key, item in dofs_dict.items():
            if key in allowed_keys:
                self.dofs.__setattr__(key, item)
            else:
                raise AttributeError(self.key_error_message(key, self.dofs))

    def set_solver(self, solver_dict: Dict) -> None:
        self.solver = Solver()
        allowed_keys = self.solver.__dict__.keys()
        for key, item in solver_dict.items():
            if key in allowed_keys:
                self.solver.__setattr__(key, item)
            else:
                raise AttributeError(self.key_error_message(key, self.solver))

    def set_materials(self, materials_list: List) -> None:
        self.materials = []
        for material_dict in materials_list:
            material = Material()
            allowed_keys = material.__dict__.keys()
            for key, item in material_dict.items():
                if key in allowed_keys:
                    material.__setattr__(key, item)
                else:
                    raise AttributeError(self.key_error_message(key, material))
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
                    raise AttributeError(self.key_error_message(key, domain))
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
                    raise AttributeError(self.key_error_message(key, bc))
            self.bcs.append(bc)

    def set_outputs(self, outputs_list: List) -> None:
        self.outputs = []
        for output_dict in outputs_list:
            output = Output()
            allowed_keys = output.__dict__.keys()
            for key, item in output_dict.items():
                if key in allowed_keys:
                    output.__setattr__(key, item)
                else:
                    raise AttributeError(self.key_error_message(key, output))
            self.outputs.append(output)

    def set_nodes_from_gmsh(self):
        self.nodes = NodeSet()
        self.nodes.read_gmsh_file(self.mesh.file)
        self.nodes.update_indices()

    def set_elements_from_gmsh(self):
        self.elements = ElementSet()
        self.elements.read_gmsh_file(self.mesh.file)

    @staticmethod
    def key_error_message(key: Any, obj: Any) -> str:
        return error_style(f'{key} is not an allowable attribute keyword of {type(obj).__name__}')

    def read_file(self, file_name: str) -> None:
        """
        读取 .toml 格式的配置文件。
        """
        with open(file_name, "rb") as f:
            toml = tomllib.load(f)
            self.set_toml(toml)

        toml_keys = self.toml.keys()
        allowed_keys = self.__dict__.keys()

        for key in toml_keys:
            if key not in allowed_keys:
                error_msg = f'{key} is not an allowable attribute keyword of {type(self).__name__}\n'
                error_msg += f'Please check the file {file_name}'
                raise AttributeError(error_style(error_msg))

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

        if 'solver' in toml_keys:
            solver_dict = self.toml['solver']
            self.set_solver(solver_dict)

        if 'outputs' in toml_keys:
            outputs_dict = self.toml['outputs']
            self.set_outputs(outputs_dict)


if __name__ == "__main__":
    props = Properties()
    # props.show()
    props.read_file(r'F:\Github\pyfem\examples\rectangle\rectangle.toml')
    props.show()
    # props.title = 1
