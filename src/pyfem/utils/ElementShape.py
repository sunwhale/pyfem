from math import sqrt

from numpy import dot, empty, zeros, cross, array, meshgrid, outer, column_stack
from numpy.polynomial.legendre import leggauss

from scipy.linalg import norm, det, inv  # type: ignore
from scipy.special import p_roots as gauss_scheme  # type: ignore
from pyfem.utils.colors import error_style

from pprint import pprint

class ShapeData:
    def __init__(self):
        self.h = None
        self.xi = None
        self.dhdxi = None
        self.x = None
        self.dhdx = None
        self.element_type = None


class ElementShapeData:

    def __init__(self):
        self.shape_data = []


class ElementShape:
    def __init__(self, element_type: str, order: int):

        if element_type == 'quad4':
            self.dimension = 2
            self.number_of_nodes = 4
            self.standard_order = 2

            ip, w = gauss_scheme(self.standard_order + order)
            xi = []
            weight = []
            for i in range(self.standard_order):
                for j in range(self.standard_order):
                    xi.append([float(ip[i].real), float(ip[j].real)])
                    weight.append(w[i] * w[j])
            pprint(xi)
            pprint(weight)

            xi, w = get_gauss_points(3, 2)
            pprint(xi)
            pprint(w)

        else:
            error_msg = f'Unsupported element type {element_type}'
            raise NotImplementedError(error_style(error_msg))

        self.element_type = element_type
        self.order = order
        self.shape_value = zeros(self.number_of_nodes)
        self.shape_gradient = zeros((self.number_of_nodes, self.dimension))
        self.jacobi_determinant = None


def get_gauss_points(dimension, order):
    xi, w = leggauss(order)
    if dimension == 1:
        pass

    elif dimension == 2:
        xi1, xi2 = meshgrid(xi, xi)
        xi1 = xi1.ravel()
        xi2 = xi2.ravel()
        xi = column_stack((xi1, xi2))

        w = outer(w, w)
        w = w.ravel()
    elif dimension == 3:
        xi1, xi2, xi3 = meshgrid(xi, xi, xi)
        xi1 = xi1.ravel()
        xi2 = xi2.ravel()
        xi3 = xi3.ravel()
        xi = column_stack((xi1, xi2, xi3))

        w = outer(outer(w, w), w)
        w = w.ravel()

    return xi, w


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
        raise NotImplementedError('2D only')

    shape_data = ShapeData()

    shape_data.h = empty(4)
    shape_data.dhdxi = empty(shape=(4, 2))
    shape_data.xi = xi

    shape_data.h[0] = 0.25 * (1.0 - xi[0]) * (1.0 - xi[1])
    shape_data.h[1] = 0.25 * (1.0 + xi[0]) * (1.0 - xi[1])
    shape_data.h[2] = 0.25 * (1.0 + xi[0]) * (1.0 + xi[1])
    shape_data.h[3] = 0.25 * (1.0 - xi[0]) * (1.0 + xi[1])

    shape_data.dhdxi[0, 0] = -0.25 * (1.0 - xi[1])
    shape_data.dhdxi[1, 0] = 0.25 * (1.0 - xi[1])
    shape_data.dhdxi[2, 0] = 0.25 * (1.0 + xi[1])
    shape_data.dhdxi[3, 0] = -0.25 * (1.0 + xi[1])

    shape_data.dhdxi[0, 1] = -0.25 * (1.0 - xi[0])
    shape_data.dhdxi[1, 1] = -0.25 * (1.0 + xi[0])
    shape_data.dhdxi[2, 1] = 0.25 * (1.0 + xi[0])
    shape_data.dhdxi[3, 1] = 0.25 * (1.0 - xi[0])

    return shape_data


def get_integration_points(element_type, order, method):
    xi = []
    weight = []

    if element_type[:-1] == "quad":
        if element_type == "quad4":
            standard_order = 2
        elif element_type == "quad8" or element_type == "quad9":
            standard_order = 3
        else:
            raise NotImplementedError('Unsupported ' + element_type)
        standard_order += order

        ip, w = gauss_scheme(standard_order)

        for i in range(standard_order):
            for j in range(standard_order):
                xi.append([float(ip[i].real), float(ip[j].real)])
                weight.append(w[i] * w[j])

    return xi, weight


def get_element_shape_data(element_coords, order=0, method='Gauss', element_type='Default'):
    element_data = ElementShapeData()

    # if element_type == 'Default':
    #     element_type = get_element_type(element_coords)

    (ip_coords, ip_wights) = get_integration_points(element_type, order, method)

    for xi, weight in zip(ip_coords, ip_wights):
        print(type(xi), xi)
        try:
            shape_data = eval('get_shape_' + element_type + '(xi)')
        except NotImplementedError:
            raise NotImplementedError('Unknown type :' + element_type)

        # print(shape_data.h)

        calc_weight_and_derivatives(element_coords, shape_data, weight)

        shape_data.x = dot(shape_data.h, element_coords)

        # print(shape_data.weight)

        element_data.shape_data.append(shape_data)

    return element_data


def calc_weight_and_derivatives(element_coords, shape_data, weight):
    jac = dot(element_coords.transpose(), shape_data.dhdxi)

    if jac.shape[0] == jac.shape[1]:
        shape_data.dhdx = dot(shape_data.dhdxi, inv(jac))
        shape_data.weight = abs(det(jac)) * weight

    elif jac.shape[0] == 2 and jac.shape[1] == 1:
        shape_data.weight = sqrt(sum(sum(jac * jac))) * weight

    elif jac.shape[0] == 3 and jac.shape[1] == 2:
        jac3 = zeros(shape=(3, 3))

        jac3[:, :2] = jac

        dA = zeros(3)

        dA[0] = norm(cross(jac3[:, 1], jac3[:, 2]))
        dA[1] = norm(cross(jac3[:, 2], jac3[:, 0]))
        dA[2] = norm(cross(jac3[:, 0], jac3[:, 1]))

        shape_data.weight = norm(dA) * weight


if __name__ == "__main__":
    from pprint import pprint

    e = get_element_shape_data(array([[0, 0], [0, 1], [1, 1], [1, 0]]), element_type='quad4')
    s = e.shape_data[0]

    es = ElementShape('quad4', 0)