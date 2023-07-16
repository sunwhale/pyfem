from typing import Callable, Dict, Tuple, get_type_hints

from numpy import empty, meshgrid, outer, column_stack, array, ndarray, insert, in1d
from numpy.polynomial.legendre import leggauss

from pyfem.elements.IsoElementDiagram import IsoElementDiagram
from pyfem.fem.constants import DTYPE
from pyfem.utils.colors import error_style
from pyfem.utils.visualization import object_slots_to_string_ndarray


class IsoElementShape:
    """
    等参单元类，设置等参单元的形函数和积分点等信息

    当前支持的单元类型 ['empty', 'line2', 'line3', 'tria3', 'tria6', 'quad4', 'quad8', 'tetra4', 'hex8', 'hex20']

    :ivar element_type: 等参单元类型
    :vartype element_type: str

    :ivar diagram: 等参单元示意图（字符串形式）
    :vartype diagram: str

    :ivar dimension: 等参单元空间维度
    :vartype dimension: int

    :ivar nodes_number: 等参单元节点数目
    :vartype nodes_number: int

    :ivar order: 等参单元插值阶次
    :vartype order: int

    :ivar shape_function: 等参单元形函数
    :vartype shape_function: Callable

    :ivar gp_number: 等参单元积分点数量
    :vartype gp_number: int

    :ivar gp_coords: 等参单元积分点坐标
    :vartype gp_coords: ndarray

    :ivar gp_weights: 等参单元积分点权重
    :vartype gp_weights: ndarray

    :ivar gp_shape_values: 等参单元积分点处形函数的值
    :vartype gp_shape_values: ndarray

    :ivar gp_shape_gradients: 等参单元积分点处形函数对局部坐标梯度的值
    :vartype gp_shape_gradients: ndarray

    :ivar bc_surface_number: 等参单元边界面数量
    :vartype bc_surface_number: int

    :ivar bc_surface_nodes_dict: 等参单元边界面节点编号
    :vartype bc_surface_nodes_dict: Dict[str, Tuple]

    :ivar bc_surface_coord_dict: 等参单元边界面节点坐标
    :vartype bc_surface_coord_dict: Dict[str, Tuple]

    :ivar bc_gp_coords_dict: 等参单元边界面积分点坐标
    :vartype bc_gp_coords_dict: Dict[str, ndarray]

    :ivar bc_gp_weights: 等参单元边界面积分点权重
    :vartype bc_gp_weights: ndarray

    :ivar bc_gp_shape_values_dict: 等参单元边界面积分点处形函数的值
    :vartype bc_gp_shape_values_dict: Dict[str, ndarray]

    :ivar bc_gp_shape_gradients_dict: 等参单元边界面积分点处形函数对局部坐标梯度的值
    :vartype bc_gp_shape_gradients_dict: Dict[str, ndarray]

    :ivar nodes_to_surface_dict: 单元节点与等参单元边界面的映射字典
    :vartype nodes_to_surface_dict: Dict[str, ndarray]
    """

    __slots_dir__ = {
        'element_type': ('str', '等参单元类型'),
        'diagram': ('str', '等参单元示意图（字符串形式）'),
        'dimension': ('int', '等参单元空间维度'),
        'nodes_number': ('int', '等参单元节点数目'),
        'order': ('int', '等参单元插值阶次'),
        'shape_function': ('Callable', '等参单元形函数'),
        'gp_number': ('int', '等参单元积分点数量'),
        'gp_coords': ('ndarray', '等参单元积分点坐标'),
        'gp_weights': ('ndarray', '等参单元积分点权重'),
        'gp_shape_values': ('ndarray', '等参单元积分点处形函数的值'),
        'gp_shape_gradients': ('ndarray', '等参单元积分点处形函数对局部坐标梯度的值'),
        'bc_surface_number': ('int', '等参单元边界面数量'),
        'bc_surface_nodes_dict': ('Dict[str, Tuple]', '等参单元边界面节点编号'),
        'bc_surface_coord_dict': ('Dict[str, Tuple]', '等参单元边界面节点坐标'),
        'bc_gp_coords_dict': ('Dict[str, ndarray]', '等参单元边界面积分点坐标'),
        'bc_gp_weights': ('ndarray', '等参单元边界面积分点权重'),
        'bc_gp_shape_values_dict': ('Dict[str, ndarray]', '等参单元边界面积分点处形函数的值'),
        'bc_gp_shape_gradients_dict': ('Dict[str, ndarray]', '等参单元边界面积分点处形函数对局部坐标梯度的值'),
        'nodes_to_surface_dict': ('Dict[str, ndarray]', '单元节点与等参单元边界面的映射字典')
    }

    # for key, item in __slots_dir__.items():
    #     print(f'    :ivar {key}: {item[1]}')
    #     print(f'    :vartype {key}: {item[0]}\n')

    __slots__ = (slot for slot in __slots_dir__.keys())

    allowed_element_type = ['empty', 'line2', 'line3', 'tria3', 'tria6', 'quad4', 'quad8', 'tetra4', 'hex8', 'hex20']

    def __init__(self, element_type: str) -> None:
        self.element_type: str = ''
        self.diagram: str = ''
        self.dimension: int = 0
        self.nodes_number: int = 0
        self.order: int = 0
        self.shape_function: Callable = get_shape_empty
        self.gp_number: int = 0
        self.gp_coords: ndarray = empty(0)
        self.gp_weights: ndarray = empty(0)
        self.gp_shape_values: ndarray = empty(0)
        self.gp_shape_gradients: ndarray = empty(0)
        self.bc_surface_number: int = 0
        self.bc_surface_nodes_dict: Dict[str, Tuple] = {}
        self.bc_surface_coord_dict: Dict[str, Tuple] = {}
        self.bc_gp_coords_dict: Dict[str, ndarray] = {}
        self.bc_gp_weights: ndarray = empty(0)
        self.bc_gp_shape_values_dict: Dict[str, ndarray] = {}
        self.bc_gp_shape_gradients_dict: Dict[str, ndarray] = {}
        self.nodes_to_surface_dict: Dict[str, ndarray] = {}

        if element_type == 'empty':
            self.element_type = element_type
        elif element_type == 'line2':
            self.element_type = element_type
            self.set_line2()
        elif element_type == 'line3':
            self.element_type = element_type
            self.set_line3()
        elif element_type == 'tria3':
            self.element_type = element_type
            self.set_tria3()
        elif element_type == 'tria6':
            self.element_type = element_type
            self.set_tria6()
        elif element_type == 'quad4':
            self.element_type = element_type
            self.set_quad4()
        elif element_type == 'quad8':
            self.element_type = element_type
            self.set_quad8()
        elif element_type == 'tetra4':
            self.element_type = element_type
            self.set_tetra4()
        elif element_type == 'hex8':
            self.element_type = element_type
            self.set_hex8()
        elif element_type == 'hex20':
            self.element_type = element_type
            self.set_hex20()
        else:
            error_msg = f'Unsupported element type {element_type}'
            raise NotImplementedError(error_style(error_msg))

    def to_string(self, level: int = 1) -> str:
        return object_slots_to_string_ndarray(self, level)

    def show(self) -> None:
        print(self.to_string())

    def set_line2(self) -> None:
        self.dimension = 1
        self.nodes_number = 2
        self.order = 1
        self.gp_coords, self.gp_weights = get_gauss_points(dimension=self.dimension, order=self.order)
        self.gp_number = len(self.gp_weights)
        self.shape_function = get_shape_line2
        gp_shape_values = []
        gp_shape_gradients = []
        for gp_coord in self.gp_coords:
            h, dhdxi = self.shape_function(gp_coord)
            gp_shape_values.append(h)
            gp_shape_gradients.append(dhdxi)
        self.gp_shape_values = array(gp_shape_values)
        self.gp_shape_gradients = array(gp_shape_gradients)
        self.diagram = IsoElementDiagram.line2

    def set_line3(self) -> None:
        self.dimension = 1
        self.nodes_number = 3
        self.order = 2
        self.gp_coords, self.gp_weights = get_gauss_points(dimension=self.dimension, order=self.order)
        self.gp_number = len(self.gp_weights)
        self.shape_function = get_shape_line3
        gp_shape_values = []
        gp_shape_gradients = []
        for gp_coord in self.gp_coords:
            h, dhdxi = self.shape_function(gp_coord)
            gp_shape_values.append(h)
            gp_shape_gradients.append(dhdxi)
        self.gp_shape_values = array(gp_shape_values)
        self.gp_shape_gradients = array(gp_shape_gradients)
        self.diagram = IsoElementDiagram.line3

    def set_quad4(self) -> None:
        self.dimension = 2
        self.nodes_number = 4
        self.order = 2
        self.gp_coords, self.gp_weights = get_gauss_points(dimension=self.dimension, order=self.order)
        self.gp_number = len(self.gp_weights)
        self.shape_function = get_shape_quad4
        gp_shape_values = []
        gp_shape_gradients = []
        for gp_coord in self.gp_coords:
            h, dhdxi = self.shape_function(gp_coord)
            gp_shape_values.append(h)
            gp_shape_gradients.append(dhdxi)
        self.gp_shape_values = array(gp_shape_values)
        self.gp_shape_gradients = array(gp_shape_gradients)
        self.bc_surface_number = 4
        self.bc_surface_nodes_dict = {'s1': (3, 0),
                                      's2': (1, 2),
                                      's3': (0, 1),
                                      's4': (2, 3)}
        self.bc_surface_coord_dict = {'s1': (0, -1, -1, 1),
                                      's2': (0, 1, 1, 1),
                                      's3': (1, -1, 1, 1),
                                      's4': (1, 1, -1, 1)}
        for surface_name, surface_conn in self.bc_surface_nodes_dict.items():
            self.nodes_to_surface_dict[surface_name] = in1d(range(self.nodes_number), surface_conn)
        bc_gp_coords, self.bc_gp_weights = get_gauss_points(dimension=self.dimension - 1, order=self.order)
        self.bc_gp_coords_dict = {'s1': insert(bc_gp_coords, 0, -1, axis=1),
                                  's2': insert(bc_gp_coords, 0, 1, axis=1),
                                  's3': insert(bc_gp_coords, 1, -1, axis=1),
                                  's4': insert(bc_gp_coords, 1, 1, axis=1)}
        self.bc_gp_shape_values_dict = {}
        self.bc_gp_shape_gradients_dict = {}
        for bc_surface_name, bc_surface_gp_coords in self.bc_gp_coords_dict.items():
            bc_gp_shape_values = []
            bc_gp_shape_gradients = []
            for bc_surface_gp_coord in bc_surface_gp_coords:
                h, dhdxi = self.shape_function(bc_surface_gp_coord)
                bc_gp_shape_values.append(h)
                bc_gp_shape_gradients.append(dhdxi)
            self.bc_gp_shape_values_dict[bc_surface_name] = array(bc_gp_shape_values)
            self.bc_gp_shape_gradients_dict[bc_surface_name] = array(bc_gp_shape_gradients)
        self.diagram = IsoElementDiagram.quad4

    def set_quad8(self) -> None:
        self.dimension = 2
        self.nodes_number = 8
        self.order = 3
        self.gp_coords, self.gp_weights = get_gauss_points(dimension=self.dimension, order=self.order)
        self.gp_number = len(self.gp_weights)
        self.shape_function = get_shape_quad8
        gp_shape_values = []
        gp_shape_gradients = []
        for gp_coord in self.gp_coords:
            h, dhdxi = self.shape_function(gp_coord)
            gp_shape_values.append(h)
            gp_shape_gradients.append(dhdxi)
        self.gp_shape_values = array(gp_shape_values)
        self.gp_shape_gradients = array(gp_shape_gradients)
        self.bc_surface_number = 4
        self.bc_surface_nodes_dict = {'s1': (3, 0, 7),
                                      's2': (1, 2, 5),
                                      's3': (0, 1, 4),
                                      's4': (2, 3, 6)}
        self.bc_surface_coord_dict = {'s1': (0, -1, -1, 1),
                                      's2': (0, 1, 1, 1),
                                      's3': (1, -1, 1, 1),
                                      's4': (1, 1, -1, 1)}
        for surface_name, surface_conn in self.bc_surface_nodes_dict.items():
            self.nodes_to_surface_dict[surface_name] = in1d(range(self.nodes_number), surface_conn)
        bc_gp_coords, self.bc_gp_weights = get_gauss_points(dimension=self.dimension - 1, order=self.order)
        self.bc_gp_coords_dict = {'s1': insert(bc_gp_coords, 0, -1, axis=1),
                                  's2': insert(bc_gp_coords, 0, 1, axis=1),
                                  's3': insert(bc_gp_coords, 1, -1, axis=1),
                                  's4': insert(bc_gp_coords, 1, 1, axis=1)}
        self.bc_gp_shape_values_dict = {}
        self.bc_gp_shape_gradients_dict = {}
        for bc_surface_name, bc_surface_gp_coords in self.bc_gp_coords_dict.items():
            bc_gp_shape_values = []
            bc_gp_shape_gradients = []
            for bc_surface_gp_coord in bc_surface_gp_coords:
                h, dhdxi = self.shape_function(bc_surface_gp_coord)
                bc_gp_shape_values.append(h)
                bc_gp_shape_gradients.append(dhdxi)
            self.bc_gp_shape_values_dict[bc_surface_name] = array(bc_gp_shape_values)
            self.bc_gp_shape_gradients_dict[bc_surface_name] = array(bc_gp_shape_gradients)
        self.diagram = IsoElementDiagram.quad8

    def set_tria3(self) -> None:
        self.dimension = 2
        self.nodes_number = 3
        self.order = 1
        self.gp_coords, self.gp_weights = get_gauss_points_triangle(order=self.order)
        self.gp_number = len(self.gp_weights)
        self.shape_function = get_shape_tria3
        gp_shape_values = []
        gp_shape_gradients = []
        for gp_coord in self.gp_coords:
            h, dhdxi = self.shape_function(gp_coord)
            gp_shape_values.append(h)
            gp_shape_gradients.append(dhdxi)
        self.gp_shape_values = array(gp_shape_values)
        self.gp_shape_gradients = array(gp_shape_gradients)
        self.diagram = IsoElementDiagram.tria3

    def set_tria6(self) -> None:
        self.dimension = 2
        self.nodes_number = 6
        self.order = 3
        self.gp_coords, self.gp_weights = get_gauss_points_triangle(order=self.order)
        self.gp_number = len(self.gp_weights)
        self.shape_function = get_shape_tria6
        gp_shape_values = []
        gp_shape_gradients = []
        for gp_coord in self.gp_coords:
            h, dhdxi = self.shape_function(gp_coord)
            gp_shape_values.append(h)
            gp_shape_gradients.append(dhdxi)
        self.gp_shape_values = array(gp_shape_values)
        self.gp_shape_gradients = array(gp_shape_gradients)
        self.diagram = IsoElementDiagram.tria6

    def set_tetra4(self) -> None:
        self.dimension = 3
        self.nodes_number = 4
        self.order = 1
        self.gp_coords, self.gp_weights = get_gauss_points_tetra(order=self.order)
        self.gp_number = len(self.gp_weights)
        self.shape_function = get_shape_tetra4
        gp_shape_values = []
        gp_shape_gradients = []
        for gp_coord in self.gp_coords:
            h, dhdxi = self.shape_function(gp_coord)
            gp_shape_values.append(h)
            gp_shape_gradients.append(dhdxi)
        self.gp_shape_values = array(gp_shape_values)
        self.gp_shape_gradients = array(gp_shape_gradients)
        self.diagram = IsoElementDiagram.tetra4

    def set_hex8(self) -> None:
        self.dimension = 3
        self.nodes_number = 8
        self.order = 2
        self.gp_coords, self.gp_weights = get_gauss_points(dimension=self.dimension, order=self.order)
        self.gp_number = len(self.gp_weights)
        self.shape_function = get_shape_hex8
        gp_shape_values = []
        gp_shape_gradients = []
        for gp_coord in self.gp_coords:
            h, dhdxi = self.shape_function(gp_coord)
            gp_shape_values.append(h)
            gp_shape_gradients.append(dhdxi)
        self.gp_shape_values = array(gp_shape_values)
        self.gp_shape_gradients = array(gp_shape_gradients)
        self.bc_surface_number = 6
        self.bc_surface_nodes_dict = {'s1': (0, 3, 7, 4),
                                      's2': (1, 2, 6, 5),
                                      's3': (0, 1, 5, 4),
                                      's4': (2, 3, 6, 7),
                                      's5': (0, 1, 2, 3),
                                      's6': (4, 5, 6, 7)}
        self.bc_surface_coord_dict = {'s1': (0, -1, 1, 1),
                                      's2': (0, 1, 1, 1),
                                      's3': (1, -1, 1, 1),
                                      's4': (1, 1, 1, 1),
                                      's5': (2, -1, 1, 1),
                                      's6': (2, 1, 1, 1)}
        for surface_name, surface_conn in self.bc_surface_nodes_dict.items():
            self.nodes_to_surface_dict[surface_name] = in1d(range(self.nodes_number), surface_conn)
        bc_gp_coords, self.bc_gp_weights = get_gauss_points(dimension=self.dimension - 1, order=self.order)
        self.bc_gp_coords_dict = {'s1': insert(bc_gp_coords, 0, -1, axis=1),
                                  's2': insert(bc_gp_coords, 0, 1, axis=1),
                                  's3': insert(bc_gp_coords, 1, -1, axis=1),
                                  's4': insert(bc_gp_coords, 1, 1, axis=1),
                                  's5': insert(bc_gp_coords, 2, -1, axis=1),
                                  's6': insert(bc_gp_coords, 2, 1, axis=1)}
        self.bc_gp_shape_values_dict = {}
        self.bc_gp_shape_gradients_dict = {}
        for bc_surface_name, bc_surface_gp_coords in self.bc_gp_coords_dict.items():
            bc_gp_shape_values = []
            bc_gp_shape_gradients = []
            for bc_surface_gp_coord in bc_surface_gp_coords:
                h, dhdxi = self.shape_function(bc_surface_gp_coord)
                bc_gp_shape_values.append(h)
                bc_gp_shape_gradients.append(dhdxi)
            self.bc_gp_shape_values_dict[bc_surface_name] = array(bc_gp_shape_values)
            self.bc_gp_shape_gradients_dict[bc_surface_name] = array(bc_gp_shape_gradients)
        self.diagram = IsoElementDiagram.hex8

    def set_hex20(self) -> None:
        self.dimension = 3
        self.nodes_number = 20
        self.order = 3
        self.gp_coords, self.gp_weights = get_gauss_points(dimension=self.dimension, order=self.order)
        self.gp_number = len(self.gp_weights)
        self.shape_function = get_shape_hex20
        gp_shape_values = []
        gp_shape_gradients = []
        for gp_coord in self.gp_coords:
            h, dhdxi = self.shape_function(gp_coord)
            gp_shape_values.append(h)
            gp_shape_gradients.append(dhdxi)
        self.gp_shape_values = array(gp_shape_values)
        self.gp_shape_gradients = array(gp_shape_gradients)
        self.diagram = IsoElementDiagram.hex20


def get_shape_empty(xi: ndarray) -> Tuple[ndarray, ndarray]:
    h = empty(0)
    dhdxi = empty(shape=(0, 0))

    return h, dhdxi


def get_shape_line2(xi: ndarray) -> Tuple[ndarray, ndarray]:
    """
    两节点直线单元，节点序号及局部坐标方向如图所示::

        0---------------1
                +-->x0

    """
    if len(xi) != 1:
        raise NotImplementedError(error_style(f'coordinate {xi} must be dimension 1'))

    h = empty(2)
    dhdxi = empty(shape=(1, 2))

    h[0] = 0.5 * (1.0 - xi)
    h[1] = 0.5 * (1.0 + xi)

    dhdxi[0, 0] = -0.5
    dhdxi[0, 1] = 0.5

    return h, dhdxi


def get_shape_line3(xi: ndarray) -> Tuple[ndarray, ndarray]:
    """
    三节点直线单元，节点序号及局部坐标方向如图所示::

        0-------1-------2
                +-->x0

    """
    if len(xi) != 1:
        raise NotImplementedError(error_style(f'coordinate {xi} must be dimension 1'))

    h = empty(3)
    dhdxi = empty(shape=(1, 3))

    h[0] = 0.5 * (1.0 - xi) - 0.5 * (1.0 - xi * xi)
    h[1] = 1 - xi * xi
    h[2] = 0.5 * (1.0 + xi) - 0.5 * (1.0 - xi * xi)

    dhdxi[0, 0] = -0.5 + xi
    dhdxi[0, 1] = -2.0 * xi
    dhdxi[0, 2] = 0.5 + xi

    return h, dhdxi


def get_shape_tria3(xi: ndarray) -> Tuple[ndarray, ndarray]:
    """
    三节点三角形单元，节点序号及局部坐标方向如图所示::

        2
        * *
        *   *
        *     *
        x1      *
        |         *
        0--x0 * * * 1

    """
    if len(xi) != 2:
        raise NotImplementedError(error_style(f'coordinate {xi} must be dimension 2'))

    h = empty(3)
    dhdxi = empty(shape=(2, 3))
    xi = xi

    h[0] = 1.0 - xi[0] - xi[1]
    h[1] = xi[0]
    h[2] = xi[1]

    dhdxi[0, 0] = -1.0
    dhdxi[0, 1] = 1.0
    dhdxi[0, 2] = 0.0

    dhdxi[1, 0] = -1.0
    dhdxi[1, 1] = 0.0
    dhdxi[1, 2] = 1.0

    return h, dhdxi


def get_shape_quad4(xi: ndarray) -> Tuple[ndarray, ndarray]:
    """
    四节点四边形单元，节点序号及局部坐标方向如图所示::

        3---------------2
        |       x1      |
        |       |       |
        |       o--x0   |
        |               |
        |               |
        0---------------1

    """
    if len(xi) != 2:
        raise NotImplementedError(error_style(f'coordinate {xi} must be dimension 2'))

    h = empty(4, dtype=DTYPE)
    dhdxi = empty(shape=(2, 4), dtype=DTYPE)

    h[0] = 0.25 * (1.0 - xi[0]) * (1.0 - xi[1])
    h[1] = 0.25 * (1.0 + xi[0]) * (1.0 - xi[1])
    h[2] = 0.25 * (1.0 + xi[0]) * (1.0 + xi[1])
    h[3] = 0.25 * (1.0 - xi[0]) * (1.0 + xi[1])

    dhdxi[0, 0] = -0.25 * (1.0 - xi[1])
    dhdxi[0, 1] = 0.25 * (1.0 - xi[1])
    dhdxi[0, 2] = 0.25 * (1.0 + xi[1])
    dhdxi[0, 3] = -0.25 * (1.0 + xi[1])

    dhdxi[1, 0] = -0.25 * (1.0 - xi[0])
    dhdxi[1, 1] = -0.25 * (1.0 + xi[0])
    dhdxi[1, 2] = 0.25 * (1.0 + xi[0])
    dhdxi[1, 3] = 0.25 * (1.0 - xi[0])

    return h, dhdxi


def get_shape_quad8(xi: ndarray) -> Tuple[ndarray, ndarray]:
    """
    八节点四边形单元，节点序号及局部坐标方向如图所示::

        3-------6-------2
        |       x1      |
        |       |       |
        7       o--x0   5
        |               |
        |               |
        0-------4-------1

    """
    if len(xi) != 2:
        raise NotImplementedError(error_style(f'coordinate {xi} must be dimension 2'))

    h = empty(8)
    dhdxi = empty(shape=(2, 8))

    h[0] = -0.25 * (1.0 - xi[0]) * (1.0 - xi[1]) * (1.0 + xi[0] + xi[1])
    h[1] = -0.25 * (1.0 + xi[0]) * (1.0 - xi[1]) * (1.0 - xi[0] + xi[1])
    h[2] = -0.25 * (1.0 + xi[0]) * (1.0 + xi[1]) * (1.0 - xi[0] - xi[1])
    h[3] = -0.25 * (1.0 - xi[0]) * (1.0 + xi[1]) * (1.0 + xi[0] - xi[1])
    h[4] = 0.5 * (1.0 - xi[0]) * (1.0 + xi[0]) * (1.0 - xi[1])
    h[5] = 0.5 * (1.0 + xi[0]) * (1.0 + xi[1]) * (1.0 - xi[1])
    h[6] = 0.5 * (1.0 - xi[0]) * (1.0 + xi[0]) * (1.0 + xi[1])
    h[7] = 0.5 * (1.0 - xi[0]) * (1.0 + xi[1]) * (1.0 - xi[1])

    dhdxi[0, 0] = -0.25 * (-1.0 + xi[1]) * (2.0 * xi[0] + xi[1])
    dhdxi[0, 1] = 0.25 * (-1.0 + xi[1]) * (-2.0 * xi[0] + xi[1])
    dhdxi[0, 2] = 0.25 * (1.0 + xi[1]) * (2.0 * xi[0] + xi[1])
    dhdxi[0, 3] = -0.25 * (1.0 + xi[1]) * (-2.0 * xi[0] + xi[1])
    dhdxi[0, 4] = xi[0] * (-1.0 + xi[1])
    dhdxi[0, 5] = -0.5 * (1.0 + xi[1]) * (-1.0 + xi[1])
    dhdxi[0, 6] = -xi[0] * (1.0 + xi[1])
    dhdxi[0, 7] = 0.5 * (1.0 + xi[1]) * (-1.0 + xi[1])

    dhdxi[1, 0] = -0.25 * (-1.0 + xi[0]) * (xi[0] + 2.0 * xi[1])
    dhdxi[1, 1] = 0.25 * (1.0 + xi[0]) * (-xi[0] + 2.0 * xi[1])
    dhdxi[1, 2] = 0.25 * (1.0 + xi[0]) * (xi[0] + 2.0 * xi[1])
    dhdxi[1, 3] = -0.25 * (-1.0 + xi[0]) * (-xi[0] + 2.0 * xi[1])
    dhdxi[1, 4] = 0.5 * (1.0 + xi[0]) * (-1.0 + xi[0])
    dhdxi[1, 5] = -xi[1] * (1.0 + xi[0])
    dhdxi[1, 6] = -0.5 * (1.0 + xi[0]) * (-1.0 + xi[0])
    dhdxi[1, 7] = xi[1] * (-1.0 + xi[0])

    return h, dhdxi


def get_shape_tetra4(xi: ndarray) -> Tuple[ndarray, ndarray]:
    """
    四节点四面体单元，节点序号及局部坐标方向如图所示::

        3
        * **
        *   * *
        *     *  *
        *       *   2
        *        **  *
        x2    *     * *
        |  x1         **
        0--x0 * * * * * 1

    """
    if len(xi) != 3:
        raise NotImplementedError(error_style(f'coordinate {xi} must be dimension 3'))

    h = empty(4)
    dhdxi = empty(shape=(3, 4))

    h[0] = 1.0 - xi[0] - xi[1] - xi[2]
    h[1] = xi[0]
    h[2] = xi[1]
    h[3] = xi[2]

    dhdxi[0, 0] = -1.0
    dhdxi[0, 1] = 1.0
    dhdxi[0, 2] = 0.0
    dhdxi[0, 3] = 0.0

    dhdxi[1, 0] = -1.0
    dhdxi[1, 1] = 0.0
    dhdxi[1, 2] = 1.0
    dhdxi[1, 3] = 0.0

    dhdxi[2, 0] = -1.0
    dhdxi[2, 1] = 0.0
    dhdxi[2, 2] = 0.0
    dhdxi[2, 3] = 1.0

    return h, dhdxi


def get_shape_hex8(xi: ndarray) -> Tuple[ndarray, ndarray]:
    """
    八节点六面体单元，节点序号及局部坐标方向如图所示::

            7---------------6
           /|              /|
          / |     x2 x1   / |
         /  |     | /    /  |
        4---+-----|/----5   |
        |   |     o--x0 |   |
        |   3-----------+---2
        |  /            |  /
        | /             | /
        |/              |/
        0---------------1

    """
    if len(xi) != 3:
        raise NotImplementedError(error_style(f'coordinate {xi} must be dimension 3'))

    h = empty(8)
    dhdxi = empty(shape=(3, 8))

    h[0] = 0.125 * (1.0 - xi[0]) * (1.0 - xi[1]) * (1.0 - xi[2])
    h[1] = 0.125 * (1.0 + xi[0]) * (1.0 - xi[1]) * (1.0 - xi[2])
    h[2] = 0.125 * (1.0 + xi[0]) * (1.0 + xi[1]) * (1.0 - xi[2])
    h[3] = 0.125 * (1.0 - xi[0]) * (1.0 + xi[1]) * (1.0 - xi[2])
    h[4] = 0.125 * (1.0 - xi[0]) * (1.0 - xi[1]) * (1.0 + xi[2])
    h[5] = 0.125 * (1.0 + xi[0]) * (1.0 - xi[1]) * (1.0 + xi[2])
    h[6] = 0.125 * (1.0 + xi[0]) * (1.0 + xi[1]) * (1.0 + xi[2])
    h[7] = 0.125 * (1.0 - xi[0]) * (1.0 + xi[1]) * (1.0 + xi[2])

    dhdxi[0, 0] = -0.125 * (1.0 - xi[1]) * (1.0 - xi[2])
    dhdxi[0, 1] = 0.125 * (1.0 - xi[1]) * (1.0 - xi[2])
    dhdxi[0, 2] = 0.125 * (1.0 + xi[1]) * (1.0 - xi[2])
    dhdxi[0, 3] = -0.125 * (1.0 + xi[1]) * (1.0 - xi[2])
    dhdxi[0, 4] = -0.125 * (1.0 - xi[1]) * (1.0 + xi[2])
    dhdxi[0, 5] = 0.125 * (1.0 - xi[1]) * (1.0 + xi[2])
    dhdxi[0, 6] = 0.125 * (1.0 + xi[1]) * (1.0 + xi[2])
    dhdxi[0, 7] = -0.125 * (1.0 + xi[1]) * (1.0 + xi[2])
    dhdxi[1, 0] = -0.125 * (1.0 - xi[0]) * (1.0 - xi[2])

    dhdxi[1, 1] = -0.125 * (1.0 + xi[0]) * (1.0 - xi[2])
    dhdxi[1, 2] = 0.125 * (1.0 + xi[0]) * (1.0 - xi[2])
    dhdxi[1, 3] = 0.125 * (1.0 - xi[0]) * (1.0 - xi[2])
    dhdxi[1, 4] = -0.125 * (1.0 - xi[0]) * (1.0 + xi[2])
    dhdxi[1, 5] = -0.125 * (1.0 + xi[0]) * (1.0 + xi[2])
    dhdxi[1, 6] = 0.125 * (1.0 + xi[0]) * (1.0 + xi[2])
    dhdxi[1, 7] = 0.125 * (1.0 - xi[0]) * (1.0 + xi[2])

    dhdxi[2, 0] = -0.125 * (1.0 - xi[0]) * (1.0 - xi[1])
    dhdxi[2, 1] = -0.125 * (1.0 + xi[0]) * (1.0 - xi[1])
    dhdxi[2, 2] = -0.125 * (1.0 + xi[0]) * (1.0 + xi[1])
    dhdxi[2, 3] = -0.125 * (1.0 - xi[0]) * (1.0 + xi[1])
    dhdxi[2, 4] = 0.125 * (1.0 - xi[0]) * (1.0 - xi[1])
    dhdxi[2, 5] = 0.125 * (1.0 + xi[0]) * (1.0 - xi[1])
    dhdxi[2, 6] = 0.125 * (1.0 + xi[0]) * (1.0 + xi[1])
    dhdxi[2, 7] = 0.125 * (1.0 - xi[0]) * (1.0 + xi[1])

    return h, dhdxi


def get_shape_tria6(xi: ndarray) -> Tuple[ndarray, ndarray]:
    """
    六节点三角形单元，节点序号及局部坐标方向如图所示::

        2
        * *
        *   *
        5     4
        x1      *
        |         *
        0--x0 3 * * 1

    """
    # u = a0 + a1 * x0 + a2 * x1 + a3 * x0 * x1 + a4 * x0 * x0 +a5 * x1 *x1
    # v = b0 + b1 * x0 + b2 * x1 + b3 * x0 * x1 + b4 * x0 * x0 +b5 * x1 *x1
    if len(xi) != 2:
        raise NotImplementedError(error_style(f'coordinate {xi} must be dimension 2'))

    h = empty(6)
    dhdxi = empty(shape=(2, 6))
    xi = xi

    h[0] = - xi[0] + 2.0 * xi[0] * xi[0]
    h[1] = - xi[1] + 2.0 * xi[1] * xi[1]
    h[2] = - (1.0 - xi[0] - xi[1]) + 2.0 * (1.0 - xi[0] - xi[1]) * (1.0 - xi[0] - xi[1])
    h[3] = 4.0 * xi[0] * xi[1]
    h[4] = 4.0 * xi[1] * (1.0 - xi[0] - xi[1])
    h[5] = 4.0 * xi[0] * (1.0 - xi[0] - xi[1])

    dhdxi[0, 0] = -1.0 + 4.0 * xi[0]
    dhdxi[0, 1] = 0.0
    dhdxi[0, 2] = 1.0 + 4.0 * (-1.0) * (1.0 - xi[0] - xi[1])
    dhdxi[0, 3] = 4.0 * xi[1]
    dhdxi[0, 4] = -4.0 * xi[1]
    dhdxi[0, 5] = 4.0 * (1.0 - xi[0] - xi[1]) - 4.0 * xi[0]

    dhdxi[1, 0] = 0.0
    dhdxi[1, 1] = -1.0 + 4.0 * xi[1]
    dhdxi[1, 2] = 1.0 + 4.0 * (-1.0) * (1.0 - xi[0] - xi[1])
    dhdxi[1, 3] = 4.0 * xi[0]
    dhdxi[1, 4] = 4.0 * (1.0 - xi[0] - xi[1]) - 4.0 * xi[1]
    dhdxi[1, 5] = -4.0 * xi[0]

    return h, dhdxi


def get_shape_hex20(xi: ndarray) -> Tuple[ndarray, ndarray]:
    """
    二十节点六面体单元，节点序号及局部坐标方向如图所示::

            7-------14------6
           /|              /|
         15 |     x2 x1  13 |
         /  19    | /    /  18
        4---+---12|/----5   |
        |   |     +--x0 |   |
        |   3-------10--+---2
        16 /            17 /
        |11             | 9
        |/              |/
        0-------8-------1

    """
    # u = a0 + a1 * x0 + a2 * x1 + a3 * x2 + a4 * x0 * x0 + a5 * x1 * x1 + a6 * x2 * x2 + a7 * x0 * x1 + a8 * x1 * x2 +
    #     a9 * x2 * x0 + a10 * x0^3 + a11 * x1^3 + a12 * x2^3 + a13 * x0^2 * x1 + a14 * x1^2 * x2 + a15 * x2^2 * x0 +
    #     a16 * x0 * x1^2 + a17 * x1 * x2^2 + a18 * x2 * x0^2 + a19 * x0 * x1 * x2
    # v = b0 + b1 * x0 + b2 * x1 + b3 * x2 + b4 * x0 * x0 + b5 * x1 * x1 + b6 * x2 * x2 + b7 * x0 * x1 + b8 * x1 * x2 +
    #     b9 * x2 * x0 + b10 * x0^3 + b11 * x1^3 + b12 * x2^3 + b13 * x0^2 * x1 + b14 * x1^2 * x2 + b15 * x2^2 * x0 +
    #     b16 * x0 * x1^2 + b17 * x1 * x2^2 + b18 * x2 * x0^2 + b19 * x0 * x1 * x2
    # w = c0 + c1 * x0 + c2 * x1 + c3 * x2 + c4 * x0 * x0 + c5 * x1 * x1 + c6 * x2 * x2 + c7 * x0 * x1 + c8 * x1 * x2 +
    #     c9 * x2 * x0 + c10 * x0^3 + c11 * x1^3 + c12 * x2^3 + c13 * x0^2 * x1 + c14 * x1^2 * x2 + c15 * x2^2 * x0 +
    #     c16 * x0 * x1^2 + c17 * x1 * x2^2 + c18 * x2 * x0^2 + c19 * x0 * x1 * x2
    if len(xi) != 3:
        raise NotImplementedError(error_style(f'coordinate {xi} must be dimension 3'))

    h = empty(20)
    dhdxi = empty(shape=(3, 20))

    h[0] = 0.125 * (1.0 - xi[0]) * (1.0 - xi[1]) * (1.0 - xi[2]) * (- xi[0] - xi[1] - xi[2] - 2)
    h[1] = 0.125 * (1.0 + xi[0]) * (1.0 - xi[1]) * (1.0 - xi[2]) * (xi[0] - xi[1] - xi[2] - 2)
    h[2] = 0.125 * (1.0 + xi[0]) * (1.0 + xi[1]) * (1.0 - xi[2]) * (xi[0] + xi[1] - xi[2] - 2)
    h[3] = 0.125 * (1.0 - xi[0]) * (1.0 + xi[1]) * (1.0 - xi[2]) * (- xi[0] + xi[1] - xi[2] - 2)
    h[4] = 0.125 * (1.0 - xi[0]) * (1.0 - xi[1]) * (1.0 + xi[2]) * (- xi[0] - xi[1] + xi[2] - 2)
    h[5] = 0.125 * (1.0 + xi[0]) * (1.0 - xi[1]) * (1.0 + xi[2]) * (xi[0] - xi[1] + xi[2] - 2)
    h[6] = 0.125 * (1.0 + xi[0]) * (1.0 + xi[1]) * (1.0 + xi[2]) * (xi[0] + xi[1] + xi[2] - 2)
    h[7] = 0.125 * (1.0 - xi[0]) * (1.0 + xi[1]) * (1.0 + xi[2]) * (- xi[0] + xi[1] + xi[2] - 2)
    h[8] = 0.25 * (1.0 - xi[0] * xi[0]) * (1.0 - xi[1]) * (1.0 - xi[2])
    h[9] = 0.25 * (1.0 + xi[0]) * (1.0 - xi[1] * xi[1]) * (1.0 - xi[2])
    h[10] = 0.25 * (1.0 - xi[0] * xi[0]) * (1.0 + xi[1]) * (1.0 - xi[2])
    h[11] = 0.25 * (1.0 - xi[0]) * (1.0 - xi[1] * xi[1]) * (1.0 - xi[2])
    h[12] = 0.25 * (1.0 - xi[0] * xi[0]) * (1.0 - xi[1]) * (1.0 + xi[2])
    h[13] = 0.25 * (1.0 + xi[0]) * (1.0 - xi[1] * xi[1]) * (1.0 + xi[2])
    h[14] = 0.25 * (1.0 - xi[0] * xi[0]) * (1.0 + xi[1]) * (1.0 + xi[2])
    h[15] = 0.25 * (1.0 - xi[0]) * (1.0 - xi[1] * xi[1]) * (1.0 + xi[2])
    h[16] = 0.25 * (1.0 - xi[0]) * (1.0 - xi[1]) * (1.0 - xi[2] * xi[2])
    h[17] = 0.25 * (1.0 + xi[0]) * (1.0 - xi[1]) * (1.0 - xi[2] * xi[2])
    h[18] = 0.25 * (1.0 + xi[0]) * (1.0 + xi[1]) * (1.0 - xi[2] * xi[2])
    h[19] = 0.25 * (1.0 - xi[0]) * (1.0 + xi[1]) * (1.0 - xi[2] * xi[2])

    dhdxi[0, 0] = 0.125 * (1.0 - xi[1]) * (1.0 - xi[2])
    dhdxi[0, 1] = 0.125 * (1.0 - xi[1]) * (1.0 - xi[2])
    dhdxi[0, 2] = 0.125 * (1.0 + xi[1]) * (1.0 - xi[2])
    dhdxi[0, 3] = 0.125 * (1.0 + xi[1]) * (1.0 - xi[2])
    dhdxi[0, 4] = 0.125 * (1.0 - xi[1]) * (1.0 + xi[2])
    dhdxi[0, 5] = 0.125 * (1.0 - xi[1]) * (1.0 + xi[2])
    dhdxi[0, 6] = 0.125 * (1.0 + xi[1]) * (1.0 + xi[2])
    dhdxi[0, 7] = 0.125 * (1.0 + xi[1]) * (1.0 + xi[2])
    dhdxi[0, 8] = -0.50 * xi[0] * (1.0 - xi[1]) * (1.0 - xi[2])
    dhdxi[0, 9] = 0.25 * (1.0 - xi[1] * xi[1]) * (1.0 - xi[2])
    dhdxi[0, 10] = -0.50 * xi[0] * (1.0 + xi[1]) * (1.0 - xi[2])
    dhdxi[0, 11] = -0.25 * (1.0 - xi[1] * xi[1]) * (1.0 - xi[2])
    dhdxi[0, 12] = -0.50 * xi[0] * (1.0 - xi[1]) * (1.0 + xi[2])
    dhdxi[0, 13] = 0.25 * (1.0 - xi[1] * xi[1]) * (1.0 + xi[2])
    dhdxi[0, 14] = -0.50 * xi[0] * (1.0 + xi[1]) * (1.0 + xi[2])
    dhdxi[0, 15] = -0.25 * (1.0 - xi[1] * xi[1]) * (1.0 + xi[2])
    dhdxi[0, 16] = -0.25 * (1.0 - xi[1]) * (1.0 - xi[2] * xi[2])
    dhdxi[0, 17] = 0.25 * (1.0 - xi[1]) * (1.0 - xi[2] * xi[2])
    dhdxi[0, 18] = 0.25 * (1.0 + xi[1]) * (1.0 - xi[2] * xi[2])
    dhdxi[0, 19] = -0.25 * (1.0 + xi[1]) * (1.0 - xi[2] * xi[2])

    dhdxi[1, 0] = 0.125 * (1.0 - xi[0]) * (1.0 - xi[2])
    dhdxi[1, 1] = 0.125 * (1.0 + xi[0]) * (1.0 - xi[2])
    dhdxi[1, 2] = 0.125 * (1.0 + xi[0]) * (1.0 - xi[2])
    dhdxi[1, 3] = 0.125 * (1.0 - xi[0]) * (1.0 - xi[2])
    dhdxi[1, 4] = 0.125 * (1.0 - xi[0]) * (1.0 + xi[2])
    dhdxi[1, 5] = 0.125 * (1.0 + xi[0]) * (1.0 + xi[2])
    dhdxi[1, 6] = 0.125 * (1.0 + xi[0]) * (1.0 + xi[2])
    dhdxi[1, 7] = 0.125 * (1.0 - xi[0]) * (1.0 + xi[2])
    dhdxi[1, 8] = -0.25 * (1.0 - xi[0] * xi[0]) * (1.0 - xi[2])
    dhdxi[1, 9] = -0.50 * (1.0 + xi[0]) * xi[1] * (1.0 - xi[2])
    dhdxi[1, 10] = 0.25 * (1.0 - xi[0] * xi[0]) * (1.0 - xi[2])
    dhdxi[1, 11] = -0.50 * (1.0 - xi[0]) * xi[1] * (1.0 - xi[2])
    dhdxi[1, 12] = -0.25 * (1.0 - xi[0] * xi[0]) * (1.0 + xi[2])
    dhdxi[1, 13] = -0.50 * (1.0 + xi[0]) * xi[1] * (1.0 + xi[2])
    dhdxi[1, 14] = 0.25 * (1.0 - xi[0] * xi[0]) * (1.0 + xi[2])
    dhdxi[1, 15] = -0.50 * (1.0 - xi[0]) * (1.0 + xi[2])
    dhdxi[1, 16] = -0.25 * (1.0 - xi[0]) * (1.0 - xi[2] * xi[2])
    dhdxi[1, 17] = -0.25 * (1.0 + xi[0]) * (1.0 - xi[2] * xi[2])
    dhdxi[1, 18] = 0.25 * (1.0 + xi[0]) * (1.0 - xi[2] * xi[2])
    dhdxi[1, 19] = 0.25 * (1.0 - xi[0]) * (1.0 - xi[2] * xi[2])

    dhdxi[2, 0] = 0.125 * (1.0 - xi[0]) * (1.0 - xi[1])
    dhdxi[2, 1] = 0.125 * (1.0 + xi[0]) * (1.0 - xi[1])
    dhdxi[2, 2] = 0.125 * (1.0 + xi[0]) * (1.0 + xi[1])
    dhdxi[2, 3] = 0.125 * (1.0 - xi[0]) * (1.0 + xi[1])
    dhdxi[2, 4] = 0.125 * (1.0 - xi[0]) * (1.0 - xi[1])
    dhdxi[2, 5] = 0.125 * (1.0 + xi[0]) * (1.0 - xi[1])
    dhdxi[2, 6] = 0.125 * (1.0 + xi[0]) * (1.0 + xi[1])
    dhdxi[2, 7] = 0.125 * (1.0 - xi[0]) * (1.0 + xi[1])
    dhdxi[2, 8] = -0.25 * (1.0 - xi[0] * xi[0]) * (1.0 - xi[1])
    dhdxi[2, 9] = -0.25 * (1.0 + xi[0]) * (1.0 - xi[1] * xi[1])
    dhdxi[2, 10] = -0.25 * (1.0 - xi[0] * xi[0]) * (1.0 + xi[1])
    dhdxi[2, 11] = -0.25 * (1.0 - xi[0]) * (1.0 - xi[1] * xi[1])
    dhdxi[2, 12] = 0.25 * (1.0 - xi[0] * xi[0]) * (1.0 - xi[1])
    dhdxi[2, 13] = 0.25 * (1.0 + xi[0]) * (1.0 - xi[1] * xi[1])
    dhdxi[2, 14] = 0.25 * (1.0 - xi[0] * xi[0]) * (1.0 + xi[1])
    dhdxi[2, 15] = 0.25 * (1.0 - xi[0]) * (1.0 - xi[1] * xi[1])
    dhdxi[2, 16] = -0.50 * (1.0 - xi[0]) * (1.0 - xi[1]) * xi[2]
    dhdxi[2, 17] = -0.50 * (1.0 + xi[0]) * (1.0 - xi[1]) * xi[2]
    dhdxi[2, 18] = -0.50 * (1.0 + xi[0]) * (1.0 + xi[1]) * xi[2]
    dhdxi[2, 19] = -0.50 * (1.0 - xi[0]) * (1.0 + xi[1]) * xi[2]

    return h, dhdxi


def get_gauss_points(dimension: int, order: int) -> Tuple[ndarray, ndarray]:
    xi, weight = leggauss(order)
    if dimension == 1:
        xi = xi.reshape(len(xi), -1)
        weight = weight.reshape(len(weight), -1)

    elif dimension == 2:
        xi1, xi2 = meshgrid(xi, xi)
        xi1 = xi1.ravel()
        xi2 = xi2.ravel()
        xi = column_stack((xi1, xi2))
        weight = outer(weight, weight)
        weight = weight.ravel()

    elif dimension == 3:
        xi1, xi2, xi3 = meshgrid(xi, xi, xi)
        xi1 = xi1.ravel()
        xi2 = xi2.ravel()
        xi3 = xi3.ravel()
        xi = column_stack((xi1, xi2, xi3))
        weight = outer(outer(weight, weight), weight)
        weight = weight.ravel()

    return xi.astype(DTYPE), weight.astype(DTYPE)


def get_gauss_points_triangle(order: int) -> Tuple[ndarray, ndarray]:
    if order == 1:
        xi = [[1.0 / 3.0, 1.0 / 3.0]]
        weight = [0.5]

    elif order == 3:
        r1 = 1.0 / 6.0
        r2 = 2.0 / 3.0
        xi = [[r1, r1], [r2, r1], [r1, r2]]
        w1 = 1.0 / 6.0
        weight = [w1, w1, w1]

    elif order == 7:
        r1 = 0.5 * 0.1012865073235
        r2 = 0.5 * 0.7974269853531
        r4 = 0.5 * 0.4701420641051
        r6 = 0.0597158717898
        r7 = 1.0 / 3.0
        xi = [[r1, r1], [r2, r1], [r1, r2], [r4, r6], [r4, r4], [r6, r4], [r7, r7]]
        w1 = 0.1259391805448
        w4 = 0.1323941527885
        w7 = 0.225
        weight = [w1, w1, w1, w4, w4, w4, w7]

    else:
        raise NotImplementedError(error_style('Order must be 1, 3 or 7'))

    return array(xi, dtype=DTYPE), array(weight, dtype=DTYPE)


def get_gauss_points_tetra(order: int) -> Tuple[ndarray, ndarray]:
    if order == 1:
        third = 1.0 / 3.0
        xi = [[third, third, third]]
        weight = [0.5 * third]
    else:
        raise NotImplementedError(error_style('Only order 1 integration implemented'))

    return array(xi, dtype=DTYPE), array(weight, dtype=DTYPE)


def get_gauss_points_pyramid(order: int) -> Tuple[ndarray, ndarray]:
    if order == 1:
        xi = [[0., 0., -0.5]]
        weight = [128.0 / 27.0]
    else:
        raise NotImplementedError(error_style('Only order 1 integration implemented'))

    return array(xi, dtype=DTYPE), array(weight, dtype=DTYPE)


def get_default_element_type(node_coords: ndarray) -> str:
    num_element_nodes = node_coords.shape[0]
    dimension = node_coords.shape[1]

    if dimension == 1:
        if num_element_nodes == 2:
            return "line2"
        elif num_element_nodes == 3:
            return "line3"
        else:
            error_msg = f'No 1D element with {num_element_nodes} nodes available'
            raise NotImplementedError(error_style(error_msg))
    elif dimension == 2:
        if num_element_nodes == 3:
            return "tria3"
        elif num_element_nodes == 4:
            return "quad4"
        elif num_element_nodes == 6:
            return "tria6"
        elif num_element_nodes == 8:
            return "quad8"
        elif num_element_nodes == 9:
            return "quad9"
        else:
            error_msg = f'No 2D element with {num_element_nodes} nodes available'
            raise NotImplementedError(error_style(error_msg))
    elif dimension == 3:
        if num_element_nodes == 4:
            return "tetra4"
        elif num_element_nodes == 5:
            return "pyramid5"
        elif num_element_nodes == 6:
            return "prism6"
        elif num_element_nodes == 8:
            return "hex8"
        elif num_element_nodes == 18:
            return "prism18"
        elif num_element_nodes == 20:
            return "hex20"
        else:
            error_msg = f'No 3D element with {num_element_nodes} nodes available'
            raise NotImplementedError(error_style(error_msg))
    else:
        raise NotImplementedError(error_style(f'Unsupported dimension {dimension}'))


if __name__ == "__main__":
    # iso_element_shape = IsoElementShape('tria3')
    iso_element_shape = IsoElementShape('quad4')
    # iso_element_shape = IsoElementShape('hex8')
    # iso_element_shape = IsoElementShape('quad8')
    # iso_element_shape = IsoElementShape('tetra4')
    # iso_element_shape = IsoElementShape('line2')
    # iso_element_shape = IsoElementShape('line3')
    # iso_element_shape = IsoElementShape('empty')

    # print(iso_element_shape.bc_gp_coords_dict['s1'])
    # print(iso_element_shape.bc_gp_weights)
    # print(iso_element_shape.bc_gp_shape_values_dict['s1'])
    # print(iso_element_shape.bc_gp_shape_gradients_dict['s1'])

    # iso_element_shape.show()
