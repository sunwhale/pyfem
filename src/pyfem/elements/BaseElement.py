from numpy import zeros

from numpy import (dot, empty, abs, column_stack, array, ndarray, dtype, float64)
from scipy.linalg import norm, det, inv  # type: ignore
from pyfem.elements.IsoElementShape import IsoElementShape


class BaseElement:
    def __init__(self, iso_element_shape: IsoElementShape):
        self.iso_element_shape: IsoElementShape = iso_element_shape
        self.connectivity: ndarray = empty(0)
        self.node_coords: ndarray = empty(0)
        self.section = None
        self.dofs = None


if __name__ == "__main__":
    iso_element_shapes = {}
    iso_element_shapes['quad4'] = IsoElementShape('quad4')

    base_element_1 = BaseElement(iso_element_shapes['quad4'])
    base_element_2 = BaseElement(iso_element_shapes['quad4'])

    # print(base_element_1.iso_element_shape.to_string())
    # print(base_element_2.iso_element_shape.to_string())

    from pyfem.io.Properties import Properties
    props = Properties()
    props.read_file(r'F:\Github\pyfem\examples\rectangle\rectangle.toml')

    elements = props.elements
    nodes = props.nodes

    # print(elements.to_string(level=0))
    # print(elements)

    base_elements = []

    for element_id, element in elements.items():
        base_element = BaseElement(iso_element_shapes['quad4'])
        base_element.connectivity = element
        node_coords = []
        for node_id in element:
            node_coords.append(nodes[node_id])
        base_element.node_coords = array(node_coords)
        base_elements.append(base_element)

    base_element = base_elements[50]

    print(base_element.node_coords.transpose().shape)
    print(base_element.iso_element_shape.gp_shape_gradients.shape)
    for gp_shape_gradient in base_element.iso_element_shape.gp_shape_gradients:
        # print(gp_shape_gradient)
        jacobi = dot(base_element.node_coords.transpose(), gp_shape_gradient)
        print(jacobi)



