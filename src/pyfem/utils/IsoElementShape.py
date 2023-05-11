from typing import Tuple, Any

from numpy import (empty, meshgrid, outer, column_stack, array, ndarray, dtype, float64)
from numpy.polynomial.legendre import leggauss

from pyfem.utils.colors import error_style, BLUE, END


class IsoElementShape:
    def __init__(self, element_type: str):
        self.element_type = None
        self.dimension = None
        self.number_of_nodes = None
        self.order = None
        self.shape_function = None
        self.shape_value = None
        self.shape_gradient = None
        self.gp_coords = None
        self.gp_weight = None

        if element_type == 'quad4':
            self.element_type = element_type
            self.set_quad4()
        else:
            error_msg = f'Unsupported element type {element_type}'
            raise NotImplementedError(error_style(error_msg))

    def to_string(self, level: int = 1) -> str:
        msg = BLUE + self.__str__() + END
        msg += '\n'
        for key, item in self.__dict__.items():
            msg += '  ' * level + f'|- {key}: {item}\n'
        return msg[:-1]

    def set_quad4(self):
        self.dimension = 2
        self.number_of_nodes = 4
        self.order = 2
        self.gp_coords, self.gp_weight = get_gauss_points(dimension=self.dimension, order=self.order)
        self.shape_function = get_shape_quad4
        shape_value = []
        shape_gradient = []
        for gp_coord in self.gp_coords:
            h, dhdxi = self.shape_function(gp_coord)
            shape_value.append(h)
            shape_gradient.append(dhdxi)
        self.shape_value = array(shape_value)
        self.shape_gradient = array(shape_gradient)


def get_shape_tria3(xi):
    """
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


def get_shape_quad4(xi):
    """
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


def get_gauss_points(dimension: int, order: int) -> Tuple[ndarray[Any, dtype[float64]], ndarray[Any, dtype[float64]]]:
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


def get_gauss_points_triangle(order: int) -> Tuple[ndarray[Any, dtype[float64]], ndarray[Any, dtype[float64]]]:
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


def get_gauss_points_tetra(order: int) -> Tuple[ndarray[Any, dtype[float64]], ndarray[Any, dtype[float64]]]:
    if order == 1:
        third = 1.0 / 3.0
        xi = [[third, third, third]]
        weight = [0.5 * third]
    else:
        raise NotImplementedError(error_style('Only order 1 integration implemented'))

    return array(xi), array(weight)


def get_gauss_points_pyramid(order: int) -> Tuple[ndarray[Any, dtype[float64]], ndarray[Any, dtype[float64]]]:
    if order == 1:
        xi = [[0., 0., -0.5]]
        weight = [128.0 / 27.0]
    else:
        raise NotImplementedError(error_style('Only order 1 integration implemented'))

    return array(xi), array(weight)


if __name__ == "__main__":
    iso_element_shape = IsoElementShape('quad4')
    print(iso_element_shape.to_string())
