from typing import Optional

from pyfem.io.Material import Material
from pyfem.materials.BaseMaterial import BaseMaterial
from pyfem.materials.ElasticIsotropic import ElasticIsotropic
from pyfem.utils.colors import error_style

material_data_dict = {
    'ElasticIsotropic': ElasticIsotropic
}


def get_material_data(material: Material, dimension: int, option: Optional[str] = None) -> BaseMaterial:
    class_name = f'{material.category}{material.type}'.strip().replace(' ', '')

    if class_name in material_data_dict:
        return material_data_dict[class_name](material, dimension, option)
    else:
        error_msg = f'{class_name} material is not supported.\n'
        error_msg += f'The allowed material types are {list(material_data_dict.keys())}.'
        raise NotImplementedError(error_style(error_msg))


if __name__ == "__main__":
    from pyfem.io.Properties import Properties

    props = Properties()
    props.read_file(r'F:\Github\pyfem\examples\rectangle\rectangle.toml')

    material_data = get_material_data(props.materials[0], 3)

    print(material_data.to_string())