# -*- coding: utf-8 -*-
"""

"""
from typing import Callable

from numpy import empty, meshgrid, outer, column_stack, array, ndarray, insert, in1d
from numpy.polynomial.legendre import leggauss

from pyfem.fem.constants import DTYPE
from pyfem.isoelements.IsoElementDiagram import IsoElementDiagram
from pyfem.isoelements.shape_functions import get_shape_line2, get_shape_tetra4, get_shape_empty, get_shape_hex20, \
    get_shape_quad4, get_shape_tria3, get_shape_line3, get_shape_quad8, get_shape_tria6, get_shape_hex8
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

    :ivar bc_surface_number: 等参单元边表面数量
    :vartype bc_surface_number: int

    :ivar bc_surface_nodes_dict: 等参单元边表面节点编号
    :vartype bc_surface_nodes_dict: dict[str, tuple]

    :ivar bc_surface_coord_dict: 等参单元边表面节点坐标
    :vartype bc_surface_coord_dict: dict[str, tuple]

    :ivar bc_gp_coords_dict: 等参单元边表面积分点坐标
    :vartype bc_gp_coords_dict: dict[str, ndarray]

    :ivar bc_gp_weights: 等参单元边表面积分点权重
    :vartype bc_gp_weights: ndarray

    :ivar bc_gp_shape_values_dict: 等参单元边表面积分点处形函数的值
    :vartype bc_gp_shape_values_dict: dict[str, ndarray]

    :ivar bc_gp_shape_gradients_dict: 等参单元边表面积分点处形函数对局部坐标梯度的值
    :vartype bc_gp_shape_gradients_dict: dict[str, ndarray]

    :ivar nodes_on_surface_dict: 单元节点与等参单元边表面的映射字典
    :vartype nodes_on_surface_dict: dict[str, ndarray]
    """

    __slots_dict__: dict = {
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
        'bc_surface_number': ('int', '等参单元边表面数量'),
        'bc_surface_nodes_dict': ('dict[str, tuple]', '等参单元边表面节点编号'),
        'bc_surface_coord_dict': ('dict[str, tuple]', '等参单元边表面节点坐标'),
        'bc_gp_coords_dict': ('dict[str, ndarray]', '等参单元边表面积分点坐标'),
        'bc_gp_weights': ('ndarray', '等参单元边表面积分点权重'),
        'bc_gp_shape_values_dict': ('dict[str, ndarray]', '等参单元边表面积分点处形函数的值'),
        'bc_gp_shape_gradients_dict': ('dict[str, ndarray]', '等参单元边表面积分点处形函数对局部坐标梯度的值'),
        'nodes_on_surface_dict': ('dict[str, ndarray]', '单元节点与等参单元边表面的映射字典')
    }

    __slots__: list = [slot for slot in __slots_dict__.keys()]

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
        self.bc_surface_nodes_dict: dict[str, tuple] = dict()
        self.bc_surface_coord_dict: dict[str, tuple] = dict()
        self.bc_gp_coords_dict: dict[str, ndarray] = dict()
        self.bc_gp_weights: ndarray = empty(0)
        self.bc_gp_shape_values_dict: dict[str, ndarray] = dict()
        self.bc_gp_shape_gradients_dict: dict[str, ndarray] = dict()
        self.nodes_on_surface_dict: dict[str, ndarray] = dict()

        element_type_dict = {
            'line2': self.set_line2,
            'line3': self.set_line3,
            'tria3': self.set_tria3,
            'tria6': self.set_tria6,
            'quad4': self.set_quad4,
            'quad8': self.set_quad8,
            'tetra4': self.set_tetra4,
            'hex8': self.set_hex8,
            'hex20': self.set_hex20
        }

        self.element_type = element_type

        if element_type == 'empty':
            pass
        elif element_type in element_type_dict.keys():
            element_type_dict[element_type]()
        else:
            error_msg = f'unsupported element type {element_type}'
            raise NotImplementedError(error_style(error_msg))

        # 根据权重数组计算积分点数量
        self.gp_number = len(self.gp_weights)

        # 根据等参单元形函数，计算积分点处形函数的值和形函数梯度的值
        gp_shape_values = list()
        gp_shape_gradients = list()
        for gp_coord in self.gp_coords:
            N, dNdxi = self.shape_function(gp_coord)
            gp_shape_values.append(N)
            gp_shape_gradients.append(dNdxi)
        self.gp_shape_values = array(gp_shape_values)
        self.gp_shape_gradients = array(gp_shape_gradients)

        # 建立等参单元表面名称和单元节点是否在当前表面的映射关系
        for surface_name, surface_conn in self.bc_surface_nodes_dict.items():
            self.nodes_on_surface_dict[surface_name] = array(in1d(range(self.nodes_number), surface_conn))

        # 计算等参单元表面积分点处形函数的值和形函数梯度的值
        self.bc_gp_shape_values_dict = dict()
        self.bc_gp_shape_gradients_dict = dict()
        for bc_surface_name, bc_surface_gp_coords in self.bc_gp_coords_dict.items():
            bc_gp_shape_values = list()
            bc_gp_shape_gradients = list()
            for bc_surface_gp_coord in bc_surface_gp_coords:
                N, dNdxi = self.shape_function(bc_surface_gp_coord)
                bc_gp_shape_values.append(N)
                bc_gp_shape_gradients.append(dNdxi)
            self.bc_gp_shape_values_dict[bc_surface_name] = array(bc_gp_shape_values)
            self.bc_gp_shape_gradients_dict[bc_surface_name] = array(bc_gp_shape_gradients)

    def to_string(self, level: int = 1) -> str:
        return object_slots_to_string_ndarray(self, level)

    def show(self) -> None:
        print(self.to_string())

    def set_line2(self) -> None:
        self.dimension = 1
        self.nodes_number = 2
        self.order = 1
        self.gp_coords, self.gp_weights = get_gauss_points(dimension=self.dimension, order=self.order)
        self.shape_function = get_shape_line2
        self.diagram = IsoElementDiagram.line2

    def set_line3(self) -> None:
        self.dimension = 1
        self.nodes_number = 3
        self.order = 2
        self.gp_coords, self.gp_weights = get_gauss_points(dimension=self.dimension, order=self.order)
        self.shape_function = get_shape_line3
        self.diagram = IsoElementDiagram.line3

    def set_quad4(self) -> None:
        self.dimension = 2
        self.nodes_number = 4
        self.order = 2
        self.gp_coords, self.gp_weights = get_gauss_points(dimension=self.dimension, order=self.order)
        self.shape_function = get_shape_quad4

        self.bc_surface_number = 4
        bc_gp_coords, self.bc_gp_weights = get_gauss_points(dimension=self.dimension - 1, order=self.order)
        self.bc_surface_nodes_dict = {'s1': (3, 0),
                                      's2': (1, 2),
                                      's3': (0, 1),
                                      's4': (2, 3)}
        self.bc_surface_coord_dict = {'s1': (0, -1, -1, 1),
                                      's2': (0, 1, 1, 1),
                                      's3': (1, -1, 1, 1),
                                      's4': (1, 1, -1, 1)}
        self.bc_gp_coords_dict = {'s1': insert(bc_gp_coords, 0, -1, axis=1),
                                  's2': insert(bc_gp_coords, 0, 1, axis=1),
                                  's3': insert(bc_gp_coords, 1, -1, axis=1),
                                  's4': insert(bc_gp_coords, 1, 1, axis=1)}
        self.diagram = IsoElementDiagram.quad4

    def set_quad8(self) -> None:
        self.dimension = 2
        self.nodes_number = 8
        self.order = 3
        self.gp_coords, self.gp_weights = get_gauss_points(dimension=self.dimension, order=self.order)
        self.shape_function = get_shape_quad8

        self.bc_surface_number = 4
        bc_gp_coords, self.bc_gp_weights = get_gauss_points(dimension=self.dimension - 1, order=self.order)
        self.bc_surface_nodes_dict = {'s1': (3, 0, 7),
                                      's2': (1, 2, 5),
                                      's3': (0, 1, 4),
                                      's4': (2, 3, 6)}
        self.bc_surface_coord_dict = {'s1': (0, -1, -1, 1),
                                      's2': (0, 1, 1, 1),
                                      's3': (1, -1, 1, 1),
                                      's4': (1, 1, -1, 1)}
        self.bc_gp_coords_dict = {'s1': insert(bc_gp_coords, 0, -1, axis=1),
                                  's2': insert(bc_gp_coords, 0, 1, axis=1),
                                  's3': insert(bc_gp_coords, 1, -1, axis=1),
                                  's4': insert(bc_gp_coords, 1, 1, axis=1)}
        self.diagram = IsoElementDiagram.quad8

    def set_tria3(self) -> None:
        self.dimension = 2
        self.nodes_number = 3
        self.order = 1
        self.gp_coords, self.gp_weights = get_gauss_points_triangle(order=self.order)
        self.shape_function = get_shape_tria3
        self.diagram = IsoElementDiagram.tria3

    def set_tria6(self) -> None:
        self.dimension = 2
        self.nodes_number = 6
        self.order = 3
        self.gp_coords, self.gp_weights = get_gauss_points_triangle(order=self.order)
        self.shape_function = get_shape_tria6
        self.diagram = IsoElementDiagram.tria6

    def set_tetra4(self) -> None:
        self.dimension = 3
        self.nodes_number = 4
        self.order = 1
        self.gp_coords, self.gp_weights = get_gauss_points_tetra(order=self.order)
        self.shape_function = get_shape_tetra4
        self.diagram = IsoElementDiagram.tetra4

    def set_hex8(self) -> None:
        self.dimension = 3
        self.nodes_number = 8
        self.order = 2
        self.gp_coords, self.gp_weights = get_gauss_points(dimension=self.dimension, order=self.order)
        self.shape_function = get_shape_hex8

        self.bc_surface_number = 6
        bc_gp_coords, self.bc_gp_weights = get_gauss_points(dimension=self.dimension - 1, order=self.order)
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
        self.bc_gp_coords_dict = {'s1': insert(bc_gp_coords, 0, -1, axis=1),
                                  's2': insert(bc_gp_coords, 0, 1, axis=1),
                                  's3': insert(bc_gp_coords, 1, -1, axis=1),
                                  's4': insert(bc_gp_coords, 1, 1, axis=1),
                                  's5': insert(bc_gp_coords, 2, -1, axis=1),
                                  's6': insert(bc_gp_coords, 2, 1, axis=1)}
        self.diagram = IsoElementDiagram.hex8

    def set_hex20(self) -> None:
        self.dimension = 3
        self.nodes_number = 20
        self.order = 3
        self.gp_coords, self.gp_weights = get_gauss_points(dimension=self.dimension, order=self.order)
        self.shape_function = get_shape_hex20
        self.diagram = IsoElementDiagram.hex20


def get_gauss_points(dimension: int, order: int) -> tuple[ndarray, ndarray]:
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


def get_gauss_points_triangle(order: int) -> tuple[ndarray, ndarray]:
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


def get_gauss_points_tetra(order: int) -> tuple[ndarray, ndarray]:
    if order == 1:
        third = 1.0 / 3.0
        xi = [[third, third, third]]
        weight = [0.5 * third]
    else:
        raise NotImplementedError(error_style('Only order 1 integration implemented'))

    return array(xi, dtype=DTYPE), array(weight, dtype=DTYPE)


def get_gauss_points_pyramid(order: int) -> tuple[ndarray, ndarray]:
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
    from pyfem.utils.visualization import print_slots_dict

    print_slots_dict(IsoElementShape.__slots_dict__)

    # iso_element_shape = IsoElementShape('tria3')
    # iso_element_shape = IsoElementShape('quad4')
    # iso_element_shape = IsoElementShape('hex8')
    # iso_element_shape = IsoElementShape('quad8')
    # iso_element_shape = IsoElementShape('tetra4')
    iso_element_shape = IsoElementShape('line2')
    # iso_element_shape = IsoElementShape('line3')
    # iso_element_shape = IsoElementShape('empty')
    iso_element_shape.show()
