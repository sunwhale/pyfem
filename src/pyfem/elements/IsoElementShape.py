from typing import Tuple, Union, Callable

from numpy import (empty, meshgrid, outer, column_stack, array, ndarray, dtype, float64)
from numpy.polynomial.legendre import leggauss

from pyfem.elements.IsoElementDiagram import IsoElementDiagram
from pyfem.utils.colors import error_style
from pyfem.utils.visualization import object_dict_to_string_ndarray


class IsoElementShape:
    """
    等参单元类，设置等参单元的形函数和积分点等信息
    """
    allowed_element_type = ['empty', 'line2', 'line3', 'tria3', 'quad4', 'quad8', 'tetra4', 'hex8']

    def __init__(self, element_type: str) -> None:
        """
        当前支持的单元类型 ['empty', 'line2', 'line3', 'tria3', 'quad4', 'quad8', 'tetra4', 'hex8']
        """
        self.element_type: str = ''
        self.diagram: str = ''
        self.dimension: int = 0
        self.nodes_number: int = 0
        self.order: int = 0
        self.shape_function: Callable = get_shape_empty
        self.gp_number = 0
        self.gp_coords: ndarray = empty(0)
        self.gp_weights: ndarray = empty(0)
        self.gp_shape_values: ndarray = empty(0)
        self.gp_shape_gradients: ndarray = empty(0)

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
        else:
            error_msg = f'Unsupported element type {element_type}'
            raise NotImplementedError(error_style(error_msg))

    def to_string(self, level: int = 1) -> str:
        return object_dict_to_string_ndarray(self, level)

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
        self.diagram = IsoElementDiagram.hex8


def get_shape_empty(xi: ndarray) -> Tuple[ndarray, ndarray]:
    h = empty(0)
    dhdxi = empty(shape=(0, 0))

    return h, dhdxi


def get_shape_line2(xi: ndarray) -> Tuple[ndarray, ndarray]:
    """
    两节点直线单元
    """

    # 0---------------1
    #         +-->x0

    if len(xi) != 1:
        raise NotImplementedError(error_style(f'coordinate {xi} must be dimension 1'))

    h = empty(2)
    dhdxi = empty(shape=(2, 1))

    h[0] = 0.5 * (1.0 - xi)
    h[1] = 0.5 * (1.0 + xi)

    dhdxi[0, 0] = -0.5
    dhdxi[1, 0] = 0.5

    return h, dhdxi


def get_shape_line3(xi: ndarray) -> Tuple[ndarray, ndarray]:
    """
    三节点直线单元
    """

    # 0-------1-------2
    #         +-->x0

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
    三节点三角形单元
    """

    # 2
    # * *
    # *   *
    # *     *
    # x1      *
    # |         *
    # 0--x0 * * * 1

    if len(xi) != 2:
        raise NotImplementedError(error_style(f'coordinate {xi} must be dimension 2'))

    h = empty(3)
    dhdxi = empty(shape=(3, 2))
    xi = xi

    h[0] = 1.0 - xi[0] - xi[1]
    h[1] = xi[0]
    h[2] = xi[1]

    dhdxi[0, 0] = -1.0
    dhdxi[1, 0] = 1.0
    dhdxi[2, 0] = 0.0

    dhdxi[0, 1] = -1.0
    dhdxi[1, 1] = 0.0
    dhdxi[2, 1] = 1.0

    return h, dhdxi


def get_shape_quad4(xi: ndarray) -> Tuple[ndarray, ndarray]:
    """
    四节点四边形单元
    """

    # 3---------------2
    # |       x1      |
    # |       |       |
    # |       o--x0   |
    # |               |
    # |               |
    # 0---------------1

    if len(xi) != 2:
        raise NotImplementedError(error_style(f'coordinate {xi} must be dimension 2'))

    h = empty(4)
    dhdxi = empty(shape=(4, 2))

    h[0] = 0.25 * (1.0 - xi[0]) * (1.0 - xi[1])
    h[1] = 0.25 * (1.0 + xi[0]) * (1.0 - xi[1])
    h[2] = 0.25 * (1.0 + xi[0]) * (1.0 + xi[1])
    h[3] = 0.25 * (1.0 - xi[0]) * (1.0 + xi[1])

    dhdxi[0, 0] = -0.25 * (1.0 - xi[1])
    dhdxi[1, 0] = 0.25 * (1.0 - xi[1])
    dhdxi[2, 0] = 0.25 * (1.0 + xi[1])
    dhdxi[3, 0] = -0.25 * (1.0 + xi[1])

    dhdxi[0, 1] = -0.25 * (1.0 - xi[0])
    dhdxi[1, 1] = -0.25 * (1.0 + xi[0])
    dhdxi[2, 1] = 0.25 * (1.0 + xi[0])
    dhdxi[3, 1] = 0.25 * (1.0 - xi[0])

    return h, dhdxi


def get_shape_quad8(xi: ndarray) -> Tuple[ndarray, ndarray]:
    """
    八节点四边形单元
    """

    # 3-------6-------2
    # |       x1      |
    # |       |       |
    # 7       o--x0   5
    # |               |
    # |               |
    # 0-------4-------1

    if len(xi) != 2:
        raise NotImplementedError(error_style(f'coordinate {xi} must be dimension 2'))

    h = empty(8)
    dhdxi = empty(shape=(8, 2))

    h[0] = -0.25 * (1.0 - xi[0]) * (1.0 - xi[1]) * (1.0 + xi[0] + xi[1])
    h[1] = 0.5 * (1.0 - xi[0]) * (1.0 + xi[0]) * (1.0 - xi[1])
    h[2] = -0.25 * (1.0 + xi[0]) * (1.0 - xi[1]) * (1.0 - xi[0] + xi[1])
    h[3] = 0.5 * (1.0 + xi[0]) * (1.0 + xi[1]) * (1.0 - xi[1])
    h[4] = -0.25 * (1.0 + xi[0]) * (1.0 + xi[1]) * (1.0 - xi[0] - xi[1])
    h[5] = 0.5 * (1.0 - xi[0]) * (1.0 + xi[0]) * (1.0 + xi[1])
    h[6] = -0.25 * (1.0 - xi[0]) * (1.0 + xi[1]) * (1.0 + xi[0] - xi[1])
    h[7] = 0.5 * (1.0 - xi[0]) * (1.0 + xi[1]) * (1.0 - xi[1])

    dhdxi[0, 0] = -0.25 * (-1.0 + xi[1]) * (2.0 * xi[0] + xi[1])
    dhdxi[1, 0] = xi[0] * (-1.0 + xi[1])
    dhdxi[2, 0] = 0.25 * (-1.0 + xi[1]) * (-2.0 * xi[0] + xi[1])
    dhdxi[3, 0] = -0.5 * (1.0 + xi[1]) * (-1.0 + xi[1])
    dhdxi[4, 0] = 0.25 * (1.0 + xi[1]) * (2.0 * xi[0] + xi[1])
    dhdxi[5, 0] = -xi[0] * (1.0 + xi[1])
    dhdxi[6, 0] = -0.25 * (1.0 + xi[1]) * (-2.0 * xi[0] + xi[1])
    dhdxi[7, 0] = 0.5 * (1.0 + xi[1]) * (-1.0 + xi[1])

    dhdxi[0, 1] = -0.25 * (-1.0 + xi[0]) * (xi[0] + 2.0 * xi[1])
    dhdxi[1, 1] = 0.5 * (1.0 + xi[0]) * (-1.0 + xi[0])
    dhdxi[2, 1] = 0.25 * (1.0 + xi[0]) * (-xi[0] + 2.0 * xi[1])
    dhdxi[3, 1] = -xi[1] * (1.0 + xi[0])
    dhdxi[4, 1] = 0.25 * (1.0 + xi[0]) * (xi[0] + 2.0 * xi[1])
    dhdxi[5, 1] = -0.5 * (1.0 + xi[0]) * (-1.0 + xi[0])
    dhdxi[6, 1] = -0.25 * (-1.0 + xi[0]) * (-xi[0] + 2.0 * xi[1])
    dhdxi[7, 1] = xi[1] * (-1.0 + xi[0])

    return h, dhdxi


def get_shape_tetra4(xi: ndarray) -> Tuple[ndarray, ndarray]:
    """
    四节点四面体单元
    """

    # 3
    # * **
    # *   * *
    # *     *  *
    # *       *   2
    # *        **  *
    # x2    *     * *
    # |  x1         **
    # 0--x0 * * * * * 1

    if len(xi) != 3:
        raise NotImplementedError(error_style(f'coordinate {xi} must be dimension 3'))

    h = empty(4)
    dhdxi = empty(shape=(4, 3))

    h[0] = 1.0 - xi[0] - xi[1] - xi[2]
    h[1] = xi[0]
    h[2] = xi[1]
    h[3] = xi[2]

    dhdxi[0, 0] = -1.0
    dhdxi[1, 0] = 1.0
    dhdxi[2, 0] = 0.0
    dhdxi[3, 0] = 0.0

    dhdxi[0, 1] = -1.0
    dhdxi[1, 1] = 0.0
    dhdxi[2, 1] = 1.0
    dhdxi[3, 1] = 0.0

    dhdxi[0, 2] = -1.0
    dhdxi[1, 2] = 0.0
    dhdxi[2, 2] = 0.0
    dhdxi[3, 2] = 1.0

    return h, dhdxi


def get_shape_hex8(xi: ndarray) -> Tuple[ndarray, ndarray]:
    """
    八节点六面体单元
    """

    #     7---------------6
    #    /|              /|
    #   / |     x2 x1   / |
    #  /  |     | /    /  |
    # 4---+-----|/----5   |
    # |   |     o--x0 |   |
    # |   3-----------+---2
    # |  /            |  /
    # | /             | /
    # |/              |/
    # 0---------------1

    if len(xi) != 3:
        raise NotImplementedError(error_style(f'coordinate {xi} must be dimension 3'))

    h = empty(8)
    dhdxi = empty(shape=(8, 3))

    h[0] = 0.125 * (1.0 - xi[0]) * (1.0 - xi[1]) * (1.0 - xi[2])
    h[1] = 0.125 * (1.0 + xi[0]) * (1.0 - xi[1]) * (1.0 - xi[2])
    h[2] = 0.125 * (1.0 + xi[0]) * (1.0 + xi[1]) * (1.0 - xi[2])
    h[3] = 0.125 * (1.0 - xi[0]) * (1.0 + xi[1]) * (1.0 - xi[2])
    h[4] = 0.125 * (1.0 - xi[0]) * (1.0 - xi[1]) * (1.0 + xi[2])
    h[5] = 0.125 * (1.0 + xi[0]) * (1.0 - xi[1]) * (1.0 + xi[2])
    h[6] = 0.125 * (1.0 + xi[0]) * (1.0 + xi[1]) * (1.0 + xi[2])
    h[7] = 0.125 * (1.0 - xi[0]) * (1.0 + xi[1]) * (1.0 + xi[2])

    dhdxi[0, 0] = -0.125 * (1.0 - xi[1]) * (1.0 - xi[2])
    dhdxi[1, 0] = 0.125 * (1.0 - xi[1]) * (1.0 - xi[2])
    dhdxi[2, 0] = 0.125 * (1.0 + xi[1]) * (1.0 - xi[2])
    dhdxi[3, 0] = -0.125 * (1.0 + xi[1]) * (1.0 - xi[2])
    dhdxi[4, 0] = -0.125 * (1.0 - xi[1]) * (1.0 + xi[2])
    dhdxi[5, 0] = 0.125 * (1.0 - xi[1]) * (1.0 + xi[2])
    dhdxi[6, 0] = 0.125 * (1.0 + xi[1]) * (1.0 + xi[2])
    dhdxi[7, 0] = -0.125 * (1.0 + xi[1]) * (1.0 + xi[2])

    dhdxi[0, 1] = -0.125 * (1.0 - xi[0]) * (1.0 - xi[2])
    dhdxi[1, 1] = -0.125 * (1.0 + xi[0]) * (1.0 - xi[2])
    dhdxi[2, 1] = 0.125 * (1.0 + xi[0]) * (1.0 - xi[2])
    dhdxi[3, 1] = 0.125 * (1.0 - xi[0]) * (1.0 - xi[2])
    dhdxi[4, 1] = -0.125 * (1.0 - xi[0]) * (1.0 + xi[2])
    dhdxi[5, 1] = -0.125 * (1.0 + xi[0]) * (1.0 + xi[2])
    dhdxi[6, 1] = 0.125 * (1.0 + xi[0]) * (1.0 + xi[2])
    dhdxi[7, 1] = 0.125 * (1.0 - xi[0]) * (1.0 + xi[2])

    dhdxi[0, 2] = -0.125 * (1.0 - xi[0]) * (1.0 - xi[1])
    dhdxi[1, 2] = -0.125 * (1.0 + xi[0]) * (1.0 - xi[1])
    dhdxi[2, 2] = -0.125 * (1.0 + xi[0]) * (1.0 + xi[1])
    dhdxi[3, 2] = -0.125 * (1.0 - xi[0]) * (1.0 + xi[1])
    dhdxi[4, 2] = 0.125 * (1.0 - xi[0]) * (1.0 - xi[1])
    dhdxi[5, 2] = 0.125 * (1.0 + xi[0]) * (1.0 - xi[1])
    dhdxi[6, 2] = 0.125 * (1.0 + xi[0]) * (1.0 + xi[1])
    dhdxi[7, 2] = 0.125 * (1.0 - xi[0]) * (1.0 + xi[1])

    return h, dhdxi


def get_gauss_points(dimension: int, order: int) -> Tuple[ndarray, ndarray]:
    xi, weight = leggauss(order)
    if dimension == 1:
        xi = xi

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

    return xi, weight


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

    return array(xi), array(weight)


def get_gauss_points_tetra(order: int) -> Tuple[ndarray, ndarray]:
    if order == 1:
        third = 1.0 / 3.0
        xi = [[third, third, third]]
        weight = [0.5 * third]
    else:
        raise NotImplementedError(error_style('Only order 1 integration implemented'))

    return array(xi), array(weight)


def get_gauss_points_pyramid(order: int) -> Tuple[ndarray, ndarray]:
    if order == 1:
        xi = [[0., 0., -0.5]]
        weight = [128.0 / 27.0]
    else:
        raise NotImplementedError(error_style('Only order 1 integration implemented'))

    return array(xi), array(weight)


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
    # iso_element_shape = IsoElementShape('quad4')
    # iso_element_shape = IsoElementShape('hex8')
    # iso_element_shape = IsoElementShape('quad8')
    # iso_element_shape = IsoElementShape('tetra4')
    # iso_element_shape = IsoElementShape('line2')
    # iso_element_shape = IsoElementShape('line3')
    iso_element_shape = IsoElementShape('empty')
    print(iso_element_shape.to_string())
