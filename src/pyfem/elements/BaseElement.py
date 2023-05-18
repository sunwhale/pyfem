from typing import List, Optional

from numpy import (dot, empty, array, ndarray)
from numpy.linalg import (det, inv)

from pyfem.elements.IsoElementShape import IsoElementShape
from pyfem.io.Dof import Dof
from pyfem.io.Material import Material
from pyfem.io.Section import Section
from pyfem.utils.colors import insert_spaces, BLUE, GREEN, END
from pyfem.utils.wrappers import show_running_time


class BaseElement:
    def __init__(self, element_id: int, iso_element_shape: IsoElementShape, connectivity: ndarray,
                 node_coords: ndarray) -> None:
        self.element_id: int = element_id
        self.iso_element_shape: IsoElementShape = iso_element_shape
        self.connectivity: ndarray = connectivity
        self.assembly_conn: ndarray = empty(0)
        self.node_coords: ndarray = node_coords
        self.gp_jacobis: ndarray = empty(0)
        self.gp_jacobi_invs: ndarray = empty(0)
        self.gp_jacobi_dets: ndarray = empty(0)
        self.dof: Optional[Dof] = None
        self.dof_names: List[str] = []
        self.element_dof_ids: List[int] = []
        self.element_dof_number: int = 0
        self.material: Optional[Material] = None
        self.section: Optional[Section] = None
        self.stiffness: ndarray = empty(0)
        self.cal_jacobi()

    def to_string(self, level: int = 1) -> str:
        msg = BLUE + self.__str__() + END
        msg += '\n'
        for key, item in self.__dict__.items():
            if isinstance(item, ndarray):
                msg += '  ' * level + GREEN + f'|- {key}: ' + END + f'{type(item)} with shape = {item.shape} \n'
                msg += insert_spaces(5 + (level - 1) * 2, f'{item}') + '\n'
            else:
                msg += '  ' * level + GREEN + f'|- {key}: ' + END + f'{item}\n'
        return msg[:-1]

    def cal_jacobi(self) -> None:
        """
        通过矩阵乘法计算每个积分点上的Jacobi矩阵。
        """

        # 以下代码为采用for循环的计算方法，结构清晰，但计算效率较低
        # self.jacobi = []
        # self.jacobi_inv = []
        # self.jacobi_det = []
        # for gp_shape_gradient in self.iso_element_shape.gp_shape_gradients:
        #     jacobi = dot(self.node_coords.transpose(), gp_shape_gradient)
        #     self.jacobi.append(jacobi)
        #     self.jacobi_inv.append(inv(jacobi))
        #     self.jacobi_det.append(det(jacobi))
        # self.jacobi = array(self.jacobi)
        # self.jacobi_inv = array(self.jacobi_inv)
        # self.jacobi_det = array(self.jacobi_det)

        # 以下代码为采用numpy高维矩阵乘法的计算方法，计算效率高，但要注意矩阵维度的变化
        self.gp_jacobis = dot(self.node_coords.transpose(), self.iso_element_shape.gp_shape_gradients).swapaxes(0, 1)
        self.gp_jacobi_invs = inv(self.gp_jacobis)
        self.gp_jacobi_dets = det(self.gp_jacobis)

    def create_element_dof_ids(self) -> None:
        for node_id in self.assembly_conn:
            for dof_id, dof_name in enumerate(self.dof_names):
                self.element_dof_ids.append(node_id * len(self.dof_names) + dof_id)


@show_running_time
def main():
    iso_element_shapes = {
        'quad4': IsoElementShape('quad4'),
        'line2': IsoElementShape('line2')
    }
    from pyfem.io.Properties import Properties

    props = Properties()
    props.read_file(r'F:\Github\pyfem\examples\rectangle\rectangle.toml')

    elements = props.elements
    nodes = props.nodes

    print(elements.to_string(level=0))
    print(props.materials[0].to_string())

    base_elements = []

    for element_id, connectivity in elements.items():
        if len(connectivity) == 4:
            element_index = elements.get_indices_by_ids([element_id])[0]
            node_coords = array(nodes.get_items_by_ids(list(connectivity)))
            base_element = BaseElement(element_index, iso_element_shapes['quad4'], connectivity, node_coords)
            base_elements.append(base_element)

    print(base_elements[0].to_string())
    # print(base_elements[0].iso_element_shape.to_string())


if __name__ == "__main__":
    main()
