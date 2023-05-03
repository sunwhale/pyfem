try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from pprint import pprint


class Properties:
    def __init__(self):
        self.toml = {}
        self.allowed_keys = ['title', 'mesh', 'dofs', 'domains', 'materials', 'bcs', 'solver', 'output']
        self.title = None
        self.mesh = None
        self.dofs = None
        self.domains = None
        self.materials = None
        self.bcs = None
        self.solver = None
        self.output = None

    def read_toml(self, file_name: str) -> None:
        with open(file_name, "rb") as f:
            self.toml = tomllib.load(f)

        for prop_key, prop_item in self.toml.items():
            if prop_key == 'title':
                self.title = prop_item
            elif prop_key == 'mesh':
                self.mesh = prop_item
            elif prop_key == 'dofs':
                self.dofs = prop_item
            elif prop_key == 'domains':
                self.domains = prop_item
            elif prop_key == 'materials':
                self.materials = prop_item
            elif prop_key == 'solver':
                self.solver = prop_item
            elif prop_key == 'bcs':
                self.bcs = prop_item
            elif prop_key == 'output':
                self.output = prop_item
            else:
                print(f'Unknown key {prop_key} in the input file.')

    def print(self):
        pprint(self.toml)


if __name__ == "__main__":
    props = Properties()
    props.read_toml(r'F:\Github\pyfem\examples\rectangle\rectangle.toml')
    pprint(props.materials)
