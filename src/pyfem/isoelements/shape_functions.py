# -*- coding: utf-8 -*-
"""

"""
import numpy as np

from pyfem.fem.constants import DTYPE
from pyfem.utils.colors import error_style


def get_shape_empty(xi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    N = np.empty(0)
    dNdxi = np.empty(shape=(0, 0))

    return N, dNdxi


def get_shape_line2(xi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r"""
    两节点直线单元。

    节点序号及局部坐标方向如图所示::

        0---------------1
                +-->x0

    对应节点的形函数表达式如下：

    .. math::
        N_{ 0 } = 0.5 - 0.5 x_{0}

    .. math::
        N_{ 1 } = 0.5 x_{0} + 0.5

    """

    if len(xi) != 1:
        raise NotImplementedError(error_style(f'coordinate {xi} must be dimension 1'))

    N = np.empty(2)
    dNdxi = np.empty(shape=(1, 2))

    N[0] = 0.5 * (1.0 - xi)
    N[1] = 0.5 * (1.0 + xi)

    dNdxi[0, 0] = -0.5
    dNdxi[0, 1] = 0.5

    return N, dNdxi


def get_shape_line3(xi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r"""
    三节点直线单元。

    节点序号及局部坐标方向如图所示::

        0-------1-------2
                +-->x0

    对应节点的形函数表达式如下：

    .. math::
        N_{ 0 } = 0.5 x_{0}^{2} - 0.5 x_{0}

    .. math::
        N_{ 1 } = 1 - x_{0}^{2}

    .. math::
        N_{ 2 } = 0.5 x_{0}^{2} + 0.5 x_{0}

    """

    if len(xi) != 1:
        raise NotImplementedError(error_style(f'coordinate {xi} must be dimension 1'))

    N = np.empty(3)
    dNdxi = np.empty(shape=(1, 3))

    N[0] = 0.5 * (1.0 - xi) - 0.5 * (1.0 - xi * xi)
    N[1] = 1.0 - xi * xi
    N[2] = 0.5 * (1.0 + xi) - 0.5 * (1.0 - xi * xi)

    dNdxi[0, 0] = -0.5 + xi
    dNdxi[0, 1] = -2.0 * xi
    dNdxi[0, 2] = 0.5 + xi

    return N, dNdxi


def get_shape_tria3(xi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r"""
    三节点三角形单元。

    节点序号及局部坐标方向如图所示::

        2
        * *
        *   *
        *     *
        x1      *
        |         *
        0--x0 * * * 1

    对应节点的形函数表达式如下：

    .. math::
        N_{ 0 } = - x_{0} - x_{1} + 1.0

    .. math::
        N_{ 1 } = x_{0}

    .. math::
        N_{ 2 } = x_{1}

    """

    if len(xi) != 2:
        raise NotImplementedError(error_style(f'coordinate {xi} must be dimension 2'))

    N = np.empty(3)
    dNdxi = np.empty(shape=(2, 3))
    xi = xi

    N[0] = 1.0 - xi[0] - xi[1]
    N[1] = xi[0]
    N[2] = xi[1]

    dNdxi[0, 0] = -1.0
    dNdxi[0, 1] = 1.0
    dNdxi[0, 2] = 0.0

    dNdxi[1, 0] = -1.0
    dNdxi[1, 1] = 0.0
    dNdxi[1, 2] = 1.0

    return N, dNdxi


def get_shape_tria3_barycentric(xi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r"""
    三节点三角形单元。

    节点序号及局部坐标方向如图所示::

        2
        * *
        *   *
        *     *
        *       *
        *         *
        0 * * * * * 1

    对应节点的形函数表达式如下：

    .. math::
        N_{ 0 } = \lambda_{0}

    .. math::
        N_{ 1 } = \lambda_{1}

    .. math::
        N_{ 2 } = \lambda_{2}

    """

    if len(xi) != 3:
        raise NotImplementedError(error_style(f'barycentric coordinate {xi} must be dimension 3'))

    N = np.empty(3)
    dNdxi = np.empty(shape=(3, 3))
    xi = xi

    N[0] = xi[0]
    N[1] = xi[1]
    N[2] = xi[2]

    dNdxi[0, 0] = 1.0
    dNdxi[0, 1] = 0.0
    dNdxi[0, 2] = 0.0

    dNdxi[1, 0] = 0.0
    dNdxi[1, 1] = 1.0
    dNdxi[1, 2] = 0.0

    dNdxi[2, 0] = 0.0
    dNdxi[2, 1] = 0.0
    dNdxi[2, 2] = 1.0

    return N, dNdxi


def get_shape_quad4(xi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r"""
    四节点四边形单元。

    节点序号及局部坐标方向如图所示::

        3---------------2
        |       x1      |
        |       |       |
        |       o--x0   |
        |               |
        |               |
        0---------------1

    对应节点的形函数表达式如下：

    .. math::
        N_{ 0 } = \left(0.25 - 0.25 x_{0}\right) \left(1.0 - x_{1}\right)

    .. math::
        N_{ 1 } = \left(1.0 - x_{1}\right) \left(0.25 x_{0} + 0.25\right)

    .. math::
        N_{ 2 } = \left(0.25 x_{0} + 0.25\right) \left(x_{1} + 1.0\right)

    .. math::
        N_{ 3 } = \left(0.25 - 0.25 x_{0}\right) \left(x_{1} + 1.0\right)

    """

    if len(xi) != 2:
        raise NotImplementedError(error_style(f'coordinate {xi} must be dimension 2'))

    N = np.empty(4, dtype=DTYPE)
    dNdxi = np.empty(shape=(2, 4), dtype=DTYPE)

    N[0] = 0.25 * (1.0 - xi[0]) * (1.0 - xi[1])
    N[1] = 0.25 * (1.0 + xi[0]) * (1.0 - xi[1])
    N[2] = 0.25 * (1.0 + xi[0]) * (1.0 + xi[1])
    N[3] = 0.25 * (1.0 - xi[0]) * (1.0 + xi[1])

    dNdxi[0, 0] = -0.25 * (1.0 - xi[1])
    dNdxi[0, 1] = 0.25 * (1.0 - xi[1])
    dNdxi[0, 2] = 0.25 * (1.0 + xi[1])
    dNdxi[0, 3] = -0.25 * (1.0 + xi[1])

    dNdxi[1, 0] = -0.25 * (1.0 - xi[0])
    dNdxi[1, 1] = -0.25 * (1.0 + xi[0])
    dNdxi[1, 2] = 0.25 * (1.0 + xi[0])
    dNdxi[1, 3] = 0.25 * (1.0 - xi[0])

    return N, dNdxi


def get_shape_quad8(xi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r"""
    八节点四边形单元。

    节点序号及局部坐标方向如图所示::

        3-------6-------2
        |       x1      |
        |       |       |
        7       o--x0   5
        |               |
        |               |
        0-------4-------1

    对应节点的形函数表达式如下：

    .. math::
        N_{ 0 } = \left(1.0 - x_{1}\right) \left(0.25 x_{0} - 0.25\right) \left(x_{0} + x_{1} + 1.0\right)

    .. math::
        N_{ 1 } = \left(1.0 - x_{1}\right) \left(- 0.25 x_{0} - 0.25\right) \left(- x_{0} + x_{1} + 1.0\right)

    .. math::
        N_{ 2 } = \left(- 0.25 x_{0} - 0.25\right) \left(x_{1} + 1.0\right) \left(- x_{0} - x_{1} + 1.0\right)

    .. math::
        N_{ 3 } = \left(0.25 x_{0} - 0.25\right) \left(x_{1} + 1.0\right) \left(x_{0} - x_{1} + 1.0\right)

    .. math::
        N_{ 4 } = \left(0.5 - 0.5 x_{0}\right) \left(1.0 - x_{1}\right) \left(x_{0} + 1.0\right)

    .. math::
        N_{ 5 } = \left(1.0 - x_{1}\right) \left(0.5 x_{0} + 0.5\right) \left(x_{1} + 1.0\right)

    .. math::
        N_{ 6 } = \left(0.5 - 0.5 x_{0}\right) \left(x_{0} + 1.0\right) \left(x_{1} + 1.0\right)

    .. math::
        N_{ 7 } = \left(0.5 - 0.5 x_{0}\right) \left(1.0 - x_{1}\right) \left(x_{1} + 1.0\right)

    """

    if len(xi) != 2:
        raise NotImplementedError(error_style(f'coordinate {xi} must be dimension 2'))

    N = np.empty(8)
    dNdxi = np.empty(shape=(2, 8))

    N[0] = -0.25 * (1.0 - xi[0]) * (1.0 - xi[1]) * (1.0 + xi[0] + xi[1])
    N[1] = -0.25 * (1.0 + xi[0]) * (1.0 - xi[1]) * (1.0 - xi[0] + xi[1])
    N[2] = -0.25 * (1.0 + xi[0]) * (1.0 + xi[1]) * (1.0 - xi[0] - xi[1])
    N[3] = -0.25 * (1.0 - xi[0]) * (1.0 + xi[1]) * (1.0 + xi[0] - xi[1])
    N[4] = 0.5 * (1.0 - xi[0]) * (1.0 + xi[0]) * (1.0 - xi[1])
    N[5] = 0.5 * (1.0 + xi[0]) * (1.0 + xi[1]) * (1.0 - xi[1])
    N[6] = 0.5 * (1.0 - xi[0]) * (1.0 + xi[0]) * (1.0 + xi[1])
    N[7] = 0.5 * (1.0 - xi[0]) * (1.0 + xi[1]) * (1.0 - xi[1])

    dNdxi[0, 0] = -0.25 * (-1.0 + xi[1]) * (2.0 * xi[0] + xi[1])
    dNdxi[0, 1] = 0.25 * (-1.0 + xi[1]) * (-2.0 * xi[0] + xi[1])
    dNdxi[0, 2] = 0.25 * (1.0 + xi[1]) * (2.0 * xi[0] + xi[1])
    dNdxi[0, 3] = -0.25 * (1.0 + xi[1]) * (-2.0 * xi[0] + xi[1])
    dNdxi[0, 4] = xi[0] * (-1.0 + xi[1])
    dNdxi[0, 5] = -0.5 * (1.0 + xi[1]) * (-1.0 + xi[1])
    dNdxi[0, 6] = -xi[0] * (1.0 + xi[1])
    dNdxi[0, 7] = 0.5 * (1.0 + xi[1]) * (-1.0 + xi[1])

    dNdxi[1, 0] = -0.25 * (-1.0 + xi[0]) * (xi[0] + 2.0 * xi[1])
    dNdxi[1, 1] = 0.25 * (1.0 + xi[0]) * (-xi[0] + 2.0 * xi[1])
    dNdxi[1, 2] = 0.25 * (1.0 + xi[0]) * (xi[0] + 2.0 * xi[1])
    dNdxi[1, 3] = -0.25 * (-1.0 + xi[0]) * (-xi[0] + 2.0 * xi[1])
    dNdxi[1, 4] = 0.5 * (1.0 + xi[0]) * (-1.0 + xi[0])
    dNdxi[1, 5] = -xi[1] * (1.0 + xi[0])
    dNdxi[1, 6] = -0.5 * (1.0 + xi[0]) * (-1.0 + xi[0])
    dNdxi[1, 7] = xi[1] * (-1.0 + xi[0])

    return N, dNdxi


def get_shape_tetra4(xi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r"""
    四节点四面体单元。

    节点序号及局部坐标方向如图所示::

        3
        * **
        *   * *
        *     *  *
        *       *   2
        *        **  *
        x2    *     * *
        |  x1         **
        0--x0 * * * * * 1

    对应节点的形函数表达式如下：

    .. math::
        N_{ 0 } = - x_{0} - x_{1} - x_{2} + 1.0

    .. math::
        N_{ 1 } = x_{0}

    .. math::
        N_{ 2 } = x_{1}

    .. math::
        N_{ 3 } = x_{2}

    """

    if len(xi) != 3:
        raise NotImplementedError(error_style(f'coordinate {xi} must be dimension 3'))

    N = np.empty(4)
    dNdxi = np.empty(shape=(3, 4))

    N[0] = 1.0 - xi[0] - xi[1] - xi[2]
    N[1] = xi[0]
    N[2] = xi[1]
    N[3] = xi[2]

    dNdxi[0, 0] = -1.0
    dNdxi[0, 1] = 1.0
    dNdxi[0, 2] = 0.0
    dNdxi[0, 3] = 0.0

    dNdxi[1, 0] = -1.0
    dNdxi[1, 1] = 0.0
    dNdxi[1, 2] = 1.0
    dNdxi[1, 3] = 0.0

    dNdxi[2, 0] = -1.0
    dNdxi[2, 1] = 0.0
    dNdxi[2, 2] = 0.0
    dNdxi[2, 3] = 1.0

    return N, dNdxi


def get_shape_tetra4_barycentric(xi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r"""
    四节点四面体单元。

    节点序号及局部坐标方向如图所示::

        3
        * **
        *   * *
        *     *  *
        *       *   2
        *        **  *
        *     *     * *
        *  *          **
        0 * * * * * * * 1

    对应节点的形函数表达式如下：

    .. math::
        N_{ 0 } = - x_{0} - x_{1} - x_{2} + 1.0

    .. math::
        N_{ 1 } = x_{0}

    .. math::
        N_{ 2 } = x_{1}

    .. math::
        N_{ 3 } = x_{2}

    """

    if len(xi) != 4:
        raise NotImplementedError(error_style(f'barycentric coordinate {xi} must be dimension 4'))

    N = np.empty(4)
    dNdxi = np.empty(shape=(4, 4))

    N[0] = xi[0]
    N[1] = xi[1]
    N[2] = xi[2]
    N[3] = xi[3]

    dNdxi[0, 0] = 1.0
    dNdxi[0, 1] = 0.0
    dNdxi[0, 2] = 0.0
    dNdxi[0, 3] = 0.0

    dNdxi[1, 0] = 0.0
    dNdxi[1, 1] = 1.0
    dNdxi[1, 2] = 0.0
    dNdxi[1, 3] = 0.0

    dNdxi[2, 0] = 0.0
    dNdxi[2, 1] = 0.0
    dNdxi[2, 2] = 1.0
    dNdxi[2, 3] = 0.0

    dNdxi[3, 0] = 0.0
    dNdxi[3, 1] = 0.0
    dNdxi[3, 2] = 0.0
    dNdxi[3, 3] = 1.0

    return N, dNdxi


def get_shape_hex8(xi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r"""
    八节点六面体单元。

    节点序号及局部坐标方向如图所示::

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

    对应节点的形函数表达式如下：

    .. math::
        N_{ 0 } = \left(0.125 - 0.125 x_{0}\right) \left(1.0 - x_{1}\right) \left(1.0 - x_{2}\right)

    .. math::
        N_{ 1 } = \left(1.0 - x_{1}\right) \left(1.0 - x_{2}\right) \left(0.125 x_{0} + 0.125\right)

    .. math::
        N_{ 2 } = \left(1.0 - x_{2}\right) \left(0.125 x_{0} + 0.125\right) \left(x_{1} + 1.0\right)

    .. math::
        N_{ 3 } = \left(0.125 - 0.125 x_{0}\right) \left(1.0 - x_{2}\right) \left(x_{1} + 1.0\right)

    .. math::
        N_{ 4 } = \left(0.125 - 0.125 x_{0}\right) \left(1.0 - x_{1}\right) \left(x_{2} + 1.0\right)

    .. math::
        N_{ 5 } = \left(1.0 - x_{1}\right) \left(0.125 x_{0} + 0.125\right) \left(x_{2} + 1.0\right)

    .. math::
        N_{ 6 } = \left(0.125 x_{0} + 0.125\right) \left(x_{1} + 1.0\right) \left(x_{2} + 1.0\right)

    .. math::
        N_{ 7 } = \left(0.125 - 0.125 x_{0}\right) \left(x_{1} + 1.0\right) \left(x_{2} + 1.0\right)

    """

    if len(xi) != 3:
        raise NotImplementedError(error_style(f'coordinate {xi} must be dimension 3'))

    N = np.empty(8)
    dNdxi = np.empty(shape=(3, 8))

    N[0] = 0.125 * (1.0 - xi[0]) * (1.0 - xi[1]) * (1.0 - xi[2])
    N[1] = 0.125 * (1.0 + xi[0]) * (1.0 - xi[1]) * (1.0 - xi[2])
    N[2] = 0.125 * (1.0 + xi[0]) * (1.0 + xi[1]) * (1.0 - xi[2])
    N[3] = 0.125 * (1.0 - xi[0]) * (1.0 + xi[1]) * (1.0 - xi[2])
    N[4] = 0.125 * (1.0 - xi[0]) * (1.0 - xi[1]) * (1.0 + xi[2])
    N[5] = 0.125 * (1.0 + xi[0]) * (1.0 - xi[1]) * (1.0 + xi[2])
    N[6] = 0.125 * (1.0 + xi[0]) * (1.0 + xi[1]) * (1.0 + xi[2])
    N[7] = 0.125 * (1.0 - xi[0]) * (1.0 + xi[1]) * (1.0 + xi[2])

    dNdxi[0, 0] = -0.125 * (1.0 - xi[1]) * (1.0 - xi[2])
    dNdxi[0, 1] = 0.125 * (1.0 - xi[1]) * (1.0 - xi[2])
    dNdxi[0, 2] = 0.125 * (1.0 + xi[1]) * (1.0 - xi[2])
    dNdxi[0, 3] = -0.125 * (1.0 + xi[1]) * (1.0 - xi[2])
    dNdxi[0, 4] = -0.125 * (1.0 - xi[1]) * (1.0 + xi[2])
    dNdxi[0, 5] = 0.125 * (1.0 - xi[1]) * (1.0 + xi[2])
    dNdxi[0, 6] = 0.125 * (1.0 + xi[1]) * (1.0 + xi[2])
    dNdxi[0, 7] = -0.125 * (1.0 + xi[1]) * (1.0 + xi[2])
    dNdxi[1, 0] = -0.125 * (1.0 - xi[0]) * (1.0 - xi[2])

    dNdxi[1, 1] = -0.125 * (1.0 + xi[0]) * (1.0 - xi[2])
    dNdxi[1, 2] = 0.125 * (1.0 + xi[0]) * (1.0 - xi[2])
    dNdxi[1, 3] = 0.125 * (1.0 - xi[0]) * (1.0 - xi[2])
    dNdxi[1, 4] = -0.125 * (1.0 - xi[0]) * (1.0 + xi[2])
    dNdxi[1, 5] = -0.125 * (1.0 + xi[0]) * (1.0 + xi[2])
    dNdxi[1, 6] = 0.125 * (1.0 + xi[0]) * (1.0 + xi[2])
    dNdxi[1, 7] = 0.125 * (1.0 - xi[0]) * (1.0 + xi[2])

    dNdxi[2, 0] = -0.125 * (1.0 - xi[0]) * (1.0 - xi[1])
    dNdxi[2, 1] = -0.125 * (1.0 + xi[0]) * (1.0 - xi[1])
    dNdxi[2, 2] = -0.125 * (1.0 + xi[0]) * (1.0 + xi[1])
    dNdxi[2, 3] = -0.125 * (1.0 - xi[0]) * (1.0 + xi[1])
    dNdxi[2, 4] = 0.125 * (1.0 - xi[0]) * (1.0 - xi[1])
    dNdxi[2, 5] = 0.125 * (1.0 + xi[0]) * (1.0 - xi[1])
    dNdxi[2, 6] = 0.125 * (1.0 + xi[0]) * (1.0 + xi[1])
    dNdxi[2, 7] = 0.125 * (1.0 - xi[0]) * (1.0 + xi[1])

    return N, dNdxi


def get_shape_tria6(xi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r"""
    六节点三角形单元。

    节点序号及局部坐标方向如图所示::

        2
        * *
        *   *
        5     4
        x1      *
        |         *
        0--x0 3 * * 1

    对应节点的形函数表达式如下：

    .. math::
        N_{ 0 } = 2.0 x_{0}^{2} - x_{0}

    .. math::
        N_{ 1 } = 2.0 x_{1}^{2} - x_{1}

    .. math::
        N_{ 2 } = x_{0} + x_{1} + \left(- 2.0 x_{0} - 2.0 x_{1} + 2.0\right) \left(- x_{0} - x_{1} + 1.0\right) - 1.0

    .. math::
        N_{ 3 } = 4.0 x_{0} x_{1}

    .. math::
        N_{ 4 } = 4.0 x_{1} \left(- x_{0} - x_{1} + 1.0\right)

    .. math::
        N_{ 5 } = 4.0 x_{0} \left(- x_{0} - x_{1} + 1.0\right)

    """

    if len(xi) != 2:
        raise NotImplementedError(error_style(f'coordinate {xi} must be dimension 2'))

    N = np.empty(6)
    dNdxi = np.empty(shape=(2, 6))

    N[0] = 1.0 - xi[0] - xi[1] - 2.0 * xi[0] * (1.0 - xi[0] - xi[1]) - 2.0 * xi[1] * (1.0 - xi[0] - xi[1])
    N[1] = xi[0] - 2.0 * xi[0] * (1.0 - xi[0] - xi[1]) - 2.0 * xi[0] * xi[1]
    N[2] = xi[1] - 2.0 * xi[0] * xi[1] - 2.0 * xi[1] * (1.0 - xi[0] - xi[1])
    N[3] = 4.0 * xi[0] * (1.0 - xi[0] - xi[1])
    N[4] = 4.0 * xi[0] * xi[1]
    N[5] = 4.0 * xi[1] * (1.0 - xi[0] - xi[1])

    dNdxi[0, 0] = -1.0 - 2.0 * (1.0 - xi[0] - xi[1]) + 2.0 * xi[0] + 2.0 * xi[1]
    dNdxi[0, 1] = 1.0 - 2.0 * (1.0 - xi[0] - xi[1]) + 2.0 * xi[0] - 2.0 * xi[1]
    dNdxi[0, 2] = 0.0
    dNdxi[0, 3] = 4.0 * (1.0 - xi[0] - xi[1]) - 4.0 * xi[0]
    dNdxi[0, 4] = 4.0 * xi[1]
    dNdxi[0, 5] = -4.0 * xi[1]

    dNdxi[1, 0] = -1.0 + 2.0 * xi[0] - 2.0 * (1.0 - xi[0] - xi[1]) + 2.0 * xi[1]
    dNdxi[1, 1] = 0.0
    dNdxi[1, 2] = 1.0 - 2.0 * xi[0] - 2.0 * (1.0 - xi[0] - xi[1]) + 2.0 * xi[1]
    dNdxi[1, 3] = -4.0 * xi[0]
    dNdxi[1, 4] = 4.0 * xi[0]
    dNdxi[1, 5] = 4.0 * (1.0 - xi[0] - xi[1]) - 4.0 * xi[1]

    return N, dNdxi


def get_shape_tria6_barycentric(xi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r"""
    六节点三角形单元。

    节点序号及局部坐标方向如图所示::

        2
        * *
        *   *
        5     4
        *       *
        *         *
        0 * * 3 * * 1

    对应节点的形函数表达式如下：

    .. math::
        N_{ 0 } = 2.0 x_{0}^{2} - x_{0}

    .. math::
        N_{ 1 } = 2.0 x_{1}^{2} - x_{1}

    .. math::
        N_{ 2 } = x_{0} + x_{1} + \left(- 2.0 x_{0} - 2.0 x_{1} + 2.0\right) \left(- x_{0} - x_{1} + 1.0\right) - 1.0

    .. math::
        N_{ 3 } = 4.0 x_{0} x_{1}

    .. math::
        N_{ 4 } = 4.0 x_{1} \left(- x_{0} - x_{1} + 1.0\right)

    .. math::
        N_{ 5 } = 4.0 x_{0} \left(- x_{0} - x_{1} + 1.0\right)

    """

    if len(xi) != 3:
        raise NotImplementedError(error_style(f'barycentric coordinate {xi} must be dimension 3'))

    N = np.empty(6)
    dNdxi = np.empty(shape=(3, 6))

    N[0] = xi[0] * (2.0 * xi[0] - 1.0)
    N[1] = xi[1] * (2.0 * xi[1] - 1.0)
    N[2] = xi[2] * (2.0 * xi[2] - 1.0)
    N[3] = 4.0 * xi[0] * xi[1]
    N[4] = 4.0 * xi[1] * xi[2]
    N[5] = 4.0 * xi[2] * xi[0]

    dNdxi[0, 0] = 4.0 * xi[0] - 1.0
    dNdxi[0, 1] = 0.0
    dNdxi[0, 2] = 0.0
    dNdxi[0, 3] = 4.0 * xi[1]
    dNdxi[0, 4] = 0.0
    dNdxi[0, 5] = 4.0 * xi[2]

    dNdxi[1, 0] = 0.0
    dNdxi[1, 1] = 4.0 * xi[1] - 1.0
    dNdxi[1, 2] = 0.0
    dNdxi[1, 3] = 4.0 * xi[0]
    dNdxi[1, 4] = 4.0 * xi[2]
    dNdxi[1, 5] = 0.0

    dNdxi[2, 0] = 0.0
    dNdxi[2, 1] = 0.0
    dNdxi[2, 2] = 4.0 * xi[2] - 1.0
    dNdxi[2, 3] = 0.0
    dNdxi[2, 4] = 4.0 * xi[1]
    dNdxi[2, 5] = 4.0 * xi[0]

    return N, dNdxi


def get_shape_hex20(xi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r"""
    二十节点六面体单元形函数。

    节点序号及局部坐标方向如图所示::

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

    对应节点的形函数表达式如下：

    .. math::
        N_{ 0 } = \left(0.125 - 0.125 x_{0}\right) \left(1.0 - x_{1}\right) \left(1.0 - x_{2}\right) \left(- x_{0} - x_{1} - x_{2} - 2\right)

    .. math::
        N_{ 1 } = \left(1.0 - x_{1}\right) \left(1.0 - x_{2}\right) \left(0.125 x_{0} + 0.125\right) \left(x_{0} - x_{1} - x_{2} - 2\right)

    .. math::
        N_{ 2 } = \left(1.0 - x_{2}\right) \left(0.125 x_{0} + 0.125\right) \left(x_{1} + 1.0\right) \left(x_{0} + x_{1} - x_{2} - 2\right)

    .. math::
        N_{ 3 } = \left(0.125 - 0.125 x_{0}\right) \left(1.0 - x_{2}\right) \left(x_{1} + 1.0\right) \left(- x_{0} + x_{1} - x_{2} - 2\right)

    .. math::
        N_{ 4 } = \left(0.125 - 0.125 x_{0}\right) \left(1.0 - x_{1}\right) \left(x_{2} + 1.0\right) \left(- x_{0} - x_{1} + x_{2} - 2\right)

    .. math::
        N_{ 5 } = \left(1.0 - x_{1}\right) \left(0.125 x_{0} + 0.125\right) \left(x_{2} + 1.0\right) \left(x_{0} - x_{1} + x_{2} - 2\right)

    .. math::
        N_{ 6 } = \left(0.125 x_{0} + 0.125\right) \left(x_{1} + 1.0\right) \left(x_{2} + 1.0\right) \left(x_{0} + x_{1} + x_{2} - 2\right)

    .. math::
        N_{ 7 } = \left(0.125 - 0.125 x_{0}\right) \left(x_{1} + 1.0\right) \left(x_{2} + 1.0\right) \left(- x_{0} + x_{1} + x_{2} - 2\right)

    .. math::
        N_{ 8 } = \left(0.25 - 0.25 x_{0}^{2}\right) \left(1.0 - x_{1}\right) \left(1.0 - x_{2}\right)

    .. math::
        N_{ 9 } = \left(1.0 - x_{1}^{2}\right) \left(1.0 - x_{2}\right) \left(0.25 x_{0} + 0.25\right)

    .. math::
        N_{ 10 } = \left(0.25 - 0.25 x_{0}^{2}\right) \left(1.0 - x_{2}\right) \left(x_{1} + 1.0\right)

    .. math::
        N_{ 11 } = \left(0.25 - 0.25 x_{0}\right) \left(1.0 - x_{1}^{2}\right) \left(1.0 - x_{2}\right)

    .. math::
        N_{ 12 } = \left(0.25 - 0.25 x_{0}^{2}\right) \left(1.0 - x_{1}\right) \left(x_{2} + 1.0\right)

    .. math::
        N_{ 13 } = \left(1.0 - x_{1}^{2}\right) \left(0.25 x_{0} + 0.25\right) \left(x_{2} + 1.0\right)

    .. math::
        N_{ 14 } = \left(0.25 - 0.25 x_{0}^{2}\right) \left(x_{1} + 1.0\right) \left(x_{2} + 1.0\right)

    .. math::
        N_{ 15 } = \left(0.25 - 0.25 x_{0}\right) \left(1.0 - x_{1}^{2}\right) \left(x_{2} + 1.0\right)

    .. math::
        N_{ 16 } = \left(0.25 - 0.25 x_{0}\right) \left(1.0 - x_{1}\right) \left(1.0 - x_{2}^{2}\right)

    .. math::
        N_{ 17 } = \left(1.0 - x_{1}\right) \left(1.0 - x_{2}^{2}\right) \left(0.25 x_{0} + 0.25\right)

    .. math::
        N_{ 18 } = \left(1.0 - x_{2}^{2}\right) \left(0.25 x_{0} + 0.25\right) \left(x_{1} + 1.0\right)

    .. math::
        N_{ 19 } = \left(0.25 - 0.25 x_{0}\right) \left(1.0 - x_{2}^{2}\right) \left(x_{1} + 1.0\right)

    """

    if len(xi) != 3:
        raise NotImplementedError(error_style(f'coordinate {xi} must be dimension 3'))

    N = np.empty(20)
    dNdxi = np.empty(shape=(3, 20))

    N[0] = 0.125 * (1.0 - xi[0]) * (1.0 - xi[1]) * (1.0 - xi[2]) * (- xi[0] - xi[1] - xi[2] - 2)
    N[1] = 0.125 * (1.0 + xi[0]) * (1.0 - xi[1]) * (1.0 - xi[2]) * (xi[0] - xi[1] - xi[2] - 2)
    N[2] = 0.125 * (1.0 + xi[0]) * (1.0 + xi[1]) * (1.0 - xi[2]) * (xi[0] + xi[1] - xi[2] - 2)
    N[3] = 0.125 * (1.0 - xi[0]) * (1.0 + xi[1]) * (1.0 - xi[2]) * (- xi[0] + xi[1] - xi[2] - 2)
    N[4] = 0.125 * (1.0 - xi[0]) * (1.0 - xi[1]) * (1.0 + xi[2]) * (- xi[0] - xi[1] + xi[2] - 2)
    N[5] = 0.125 * (1.0 + xi[0]) * (1.0 - xi[1]) * (1.0 + xi[2]) * (xi[0] - xi[1] + xi[2] - 2)
    N[6] = 0.125 * (1.0 + xi[0]) * (1.0 + xi[1]) * (1.0 + xi[2]) * (xi[0] + xi[1] + xi[2] - 2)
    N[7] = 0.125 * (1.0 - xi[0]) * (1.0 + xi[1]) * (1.0 + xi[2]) * (- xi[0] + xi[1] + xi[2] - 2)
    N[8] = 0.25 * (1.0 - xi[0] * xi[0]) * (1.0 - xi[1]) * (1.0 - xi[2])
    N[9] = 0.25 * (1.0 + xi[0]) * (1.0 - xi[1] * xi[1]) * (1.0 - xi[2])
    N[10] = 0.25 * (1.0 - xi[0] * xi[0]) * (1.0 + xi[1]) * (1.0 - xi[2])
    N[11] = 0.25 * (1.0 - xi[0]) * (1.0 - xi[1] * xi[1]) * (1.0 - xi[2])
    N[12] = 0.25 * (1.0 - xi[0] * xi[0]) * (1.0 - xi[1]) * (1.0 + xi[2])
    N[13] = 0.25 * (1.0 + xi[0]) * (1.0 - xi[1] * xi[1]) * (1.0 + xi[2])
    N[14] = 0.25 * (1.0 - xi[0] * xi[0]) * (1.0 + xi[1]) * (1.0 + xi[2])
    N[15] = 0.25 * (1.0 - xi[0]) * (1.0 - xi[1] * xi[1]) * (1.0 + xi[2])
    N[16] = 0.25 * (1.0 - xi[0]) * (1.0 - xi[1]) * (1.0 - xi[2] * xi[2])
    N[17] = 0.25 * (1.0 + xi[0]) * (1.0 - xi[1]) * (1.0 - xi[2] * xi[2])
    N[18] = 0.25 * (1.0 + xi[0]) * (1.0 + xi[1]) * (1.0 - xi[2] * xi[2])
    N[19] = 0.25 * (1.0 - xi[0]) * (1.0 + xi[1]) * (1.0 - xi[2] * xi[2])

    dNdxi[(0, 0)] = 0.125 * (xi[1] - 1.0) * (xi[2] - 1.0) * (2 * xi[0] + xi[1] + xi[2] + 1)
    dNdxi[(0, 1)] = 0.125 * (xi[1] - 1.0) * (xi[2] - 1.0) * (2 * xi[0] - xi[1] - xi[2] - 1)
    dNdxi[(0, 2)] = 0.125 * (xi[1] + 1.0) * (xi[2] - 1.0) * (-2 * xi[0] - xi[1] + xi[2] + 1)
    dNdxi[(0, 3)] = 0.125 * (xi[1] + 1.0) * (xi[2] - 1.0) * (-2 * xi[0] + xi[1] - xi[2] - 1)
    dNdxi[(0, 4)] = 0.125 * (xi[1] - 1.0) * (xi[2] + 1.0) * (-2 * xi[0] - xi[1] + xi[2] - 1)
    dNdxi[(0, 5)] = 0.125 * (xi[1] - 1.0) * (xi[2] + 1.0) * (-2 * xi[0] + xi[1] - xi[2] + 1)
    dNdxi[(0, 6)] = 0.125 * (xi[1] + 1.0) * (xi[2] + 1.0) * (2 * xi[0] + xi[1] + xi[2] - 1)
    dNdxi[(0, 7)] = 0.125 * (xi[1] + 1.0) * (xi[2] + 1.0) * (2 * xi[0] - xi[1] - xi[2] + 1)
    dNdxi[(0, 8)] = -0.5 * xi[0] * (xi[1] - 1.0) * (xi[2] - 1.0)
    dNdxi[(0, 9)] = 0.25 * (xi[1] ** 2 - 1.0) * (xi[2] - 1.0)
    dNdxi[(0, 10)] = 0.5 * xi[0] * (xi[1] + 1.0) * (xi[2] - 1.0)
    dNdxi[(0, 11)] = -0.25 * (xi[1] ** 2 - 1.0) * (xi[2] - 1.0)
    dNdxi[(0, 12)] = 0.5 * xi[0] * (xi[1] - 1.0) * (xi[2] + 1.0)
    dNdxi[(0, 13)] = -0.25 * (xi[1] ** 2 - 1.0) * (xi[2] + 1.0)
    dNdxi[(0, 14)] = -0.5 * xi[0] * (xi[1] + 1.0) * (xi[2] + 1.0)
    dNdxi[(0, 15)] = 0.25 * (xi[1] ** 2 - 1.0) * (xi[2] + 1.0)
    dNdxi[(0, 16)] = -0.25 * (xi[1] - 1.0) * (xi[2] ** 2 - 1.0)
    dNdxi[(0, 17)] = 0.25 * (xi[1] - 1.0) * (xi[2] ** 2 - 1.0)
    dNdxi[(0, 18)] = -0.25 * (xi[1] + 1.0) * (xi[2] ** 2 - 1.0)
    dNdxi[(0, 19)] = 0.25 * (xi[1] + 1.0) * (xi[2] ** 2 - 1.0)
    dNdxi[(1, 0)] = 0.125 * (xi[0] - 1) * (xi[2] - 1.0) * (xi[0] + 2 * xi[1] + xi[2] + 1.0)
    dNdxi[(1, 1)] = 0.125 * (xi[0] + 1) * (xi[2] - 1.0) * (xi[0] - 2 * xi[1] - xi[2] - 1.0)
    dNdxi[(1, 2)] = 0.125 * (xi[0] + 1) * (xi[2] - 1.0) * (-xi[0] - 2 * xi[1] + xi[2] + 1.0)
    dNdxi[(1, 3)] = 0.125 * (xi[0] - 1) * (xi[2] - 1.0) * (-xi[0] + 2 * xi[1] - xi[2] - 1.0)
    dNdxi[(1, 4)] = 0.125 * (xi[0] - 1) * (xi[2] + 1.0) * (-xi[0] - 2 * xi[1] + xi[2] - 1.0)
    dNdxi[(1, 5)] = 0.125 * (xi[0] + 1) * (xi[2] + 1.0) * (-xi[0] + 2 * xi[1] - xi[2] + 1.0)
    dNdxi[(1, 6)] = 0.125 * (xi[0] + 1) * (xi[2] + 1.0) * (xi[0] + 2 * xi[1] + xi[2] - 1.0)
    dNdxi[(1, 7)] = 0.125 * (xi[0] - 1) * (xi[2] + 1.0) * (xi[0] - 2 * xi[1] - xi[2] + 1.0)
    dNdxi[(1, 8)] = -0.25 * (xi[0] ** 2 - 1) * (xi[2] - 1.0)
    dNdxi[(1, 9)] = 0.5 * xi[1] * (xi[0] + 1) * (xi[2] - 1.0)
    dNdxi[(1, 10)] = 0.25 * (xi[0] ** 2 - 1) * (xi[2] - 1.0)
    dNdxi[(1, 11)] = -0.5 * xi[1] * (xi[0] - 1) * (xi[2] - 1.0)
    dNdxi[(1, 12)] = 0.25 * (xi[0] ** 2 - 1) * (xi[2] + 1.0)
    dNdxi[(1, 13)] = -0.5 * xi[1] * (xi[0] + 1) * (xi[2] + 1.0)
    dNdxi[(1, 14)] = -0.25 * (xi[0] ** 2 - 1) * (xi[2] + 1.0)
    dNdxi[(1, 15)] = 0.5 * xi[1] * (xi[0] - 1) * (xi[2] + 1.0)
    dNdxi[(1, 16)] = -0.25 * (xi[0] - 1) * (xi[2] ** 2 - 1.0)
    dNdxi[(1, 17)] = 0.25 * (xi[0] + 1) * (xi[2] ** 2 - 1.0)
    dNdxi[(1, 18)] = -0.25 * (xi[0] + 1) * (xi[2] ** 2 - 1.0)
    dNdxi[(1, 19)] = 0.25 * (xi[0] - 1) * (xi[2] ** 2 - 1.0)
    dNdxi[(2, 0)] = 0.125 * (xi[0] - 1) * (xi[1] - 1.0) * (xi[0] + xi[1] + 2 * xi[2] + 1.0)
    dNdxi[(2, 1)] = 0.125 * (xi[0] + 1) * (xi[1] - 1.0) * (xi[0] - xi[1] - 2 * xi[2] - 1.0)
    dNdxi[(2, 2)] = 0.125 * (xi[0] + 1) * (xi[1] + 1.0) * (-xi[0] - xi[1] + 2 * xi[2] + 1.0)
    dNdxi[(2, 3)] = 0.125 * (xi[0] - 1) * (xi[1] + 1.0) * (-xi[0] + xi[1] - 2 * xi[2] - 1.0)
    dNdxi[(2, 4)] = 0.125 * (xi[0] - 1) * (xi[1] - 1.0) * (-xi[0] - xi[1] + 2 * xi[2] - 1.0)
    dNdxi[(2, 5)] = 0.125 * (xi[0] + 1) * (xi[1] - 1.0) * (-xi[0] + xi[1] - 2 * xi[2] + 1.0)
    dNdxi[(2, 6)] = 0.125 * (xi[0] + 1) * (xi[1] + 1.0) * (xi[0] + xi[1] + 2 * xi[2] - 1.0)
    dNdxi[(2, 7)] = 0.125 * (xi[0] - 1) * (xi[1] + 1.0) * (xi[0] - xi[1] - 2 * xi[2] + 1.0)
    dNdxi[(2, 8)] = -0.25 * (xi[0] ** 2 - 1) * (xi[1] - 1.0)
    dNdxi[(2, 9)] = 0.25 * (xi[0] + 1) * (xi[1] ** 2 - 1.0)
    dNdxi[(2, 10)] = 0.25 * (xi[0] ** 2 - 1) * (xi[1] + 1.0)
    dNdxi[(2, 11)] = -0.25 * (xi[0] - 1) * (xi[1] ** 2 - 1.0)
    dNdxi[(2, 12)] = 0.25 * (xi[0] ** 2 - 1) * (xi[1] - 1.0)
    dNdxi[(2, 13)] = -0.25 * (xi[0] + 1) * (xi[1] ** 2 - 1.0)
    dNdxi[(2, 14)] = -0.25 * (xi[0] ** 2 - 1) * (xi[1] + 1.0)
    dNdxi[(2, 15)] = 0.25 * (xi[0] - 1) * (xi[1] ** 2 - 1.0)
    dNdxi[(2, 16)] = -0.5 * xi[2] * (xi[0] - 1) * (xi[1] - 1.0)
    dNdxi[(2, 17)] = 0.5 * xi[2] * (xi[0] + 1) * (xi[1] - 1.0)
    dNdxi[(2, 18)] = -0.5 * xi[2] * (xi[0] + 1) * (xi[1] + 1.0)
    dNdxi[(2, 19)] = 0.5 * xi[2] * (xi[0] - 1) * (xi[1] + 1.0)

    return N, dNdxi


if __name__ == "__main__":
    # get_shape_line2(array([1]))
    print(get_shape_tria6(np.array([0, 0.5])))
