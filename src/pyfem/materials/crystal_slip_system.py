# -*- coding: utf-8 -*-
r"""
**提供晶体滑移系的信息**

本模块中包含3个字典：:py:attr:`slip_dict` ， :py:attr:`cleavage_dict` ， :py:attr:`twin_dict` 。

其中 :py:attr:`slip_dict` 字典支持的键值为::

    fcc{111}<110>
    fcc{110}<110>
    bcc{110}<111>
    bcc{112}<111>
    bcc{123}<111>
    hcp{001}<-1-10>
    hcp{1-10}<-1-10>
    hcp{-111}<-1-10>
    hcp{-101}<113>
    hcp{-112}<113>
    bct{100}<001>
    bct{110}<001>
    bct{100}<010>
    bct{110}<1-11>
    bct{110}<1-10>
    bct{100}<011>
    bct{001}<010>
    bct{001}<110>
    bct{011}<01-1>
    bct{011}<1-11>
    bct{011}<100>
    bct{211}<01-1>
    bct{211}<-111>

当晶体类型为FCC（）、BCC（）、BCT（）时，存储在字典的返回值为滑移方向和滑移面法向组成的元组 (ndarray, ndarray)。

对于HCP材料，通常其滑移面和滑移方向都是在非正交的四轴坐标系或三轴坐标系中用密勒指数表示。但在有限元软件中一般使用笛卡尔直角坐标系，所以就有必要将密勒指数转换到笛卡尔直角坐标系中。该过程分为两步进行：

1. 将四轴坐标系中的晶向指数与晶面指数转化到三轴坐标系中：
设四轴坐标系中晶向指数为[u v t w]，平面指数为[h k i l]，得到：
三轴坐标系中晶向指数[u1 v1 w1]为：

[u1 v1 w1] = [2u+v u+2v w]

三轴坐标系中晶面指数[h0 h0 h0]为：

[h0 k0 l0] = [h k l]

由于六方系的晶面指数和晶面法向指数不同，根据倒易点阵的相关知识，晶面法向指数[h1 k1 l1]为：
[h1 k1 l1] = [4h/3+k2/3 h2/3+k4/3 l(a/c)^2]

2. 将三轴坐标系中的晶向指数[u1 v1 w1]和晶面法向指数[h1 k1 l1]转换到直角坐标系，其转化关系如下图所示:
分别将三轴坐标系的a1轴与直角坐标系的x轴固定，三轴坐标系的c轴与直角坐标系的z轴固定，固定之后，y轴与a2轴之间的夹角为30度。
其中还用到了晶轴比 r = c/a，完成三轴坐标系c轴与直角坐标系的z轴长度缩放。得到转换公式：

x = a1 - a2*sin(30)  y = a2 * cos(30)  z = r * c

综合第一步的晶向指数[u1 v1 w1]和晶面法向指数[h1 k1 l1]，最后获得直角坐标系中的晶向指数[u2 v2 w2]和晶面法向指数[h2 k2 l2]为:

[u2 v2 w2] = [3u/2 (u+2v)*sqrt(3)/2 w*(c/a)]

[h2 k2 l2] = [h (h+2k)/sqrt(3) l/(c/a)]


六方晶体的三轴坐标系::

    c
    |    a2
    |   /
    |  /
    | /
    |/
    o--------------- a1


六方晶体的直角坐标系::

        z
        |
        |
        |
        |
        |
        o---------------y
       /
      /
     /
    x

整体过程：

direction [uvtw]->[2u+v u+2v w]->[3u/2 (u+2v)*sqrt(3)/2 w*(c/a)]

plane (hkil)->(hkl)>(4/3h+2/3k 2/3h+4/3k l(a/c)^2)->(h (h+2k)/sqrt(3) l/(c/a))


用符号 :math:`\left( {\square ,\square } \right)` 表示两个矢量的内积。

.. math::
    \left( {{\mathbf{a}},{{\mathbf{a}}^*}} \right) = 1

补充：
有两种点阵，a，b，c是真实点阵（正点阵）的点阵参数；a^*, b^*, c^*为前者的倒易点阵的点阵参数。它们存在以下关系：

dot(a, a^*) = 1, dot(b, b^*) = 1, dot(c*, c^*) = 1

dot(a, b^*) = dot(a, c^*) = dot(b, c^*) = dot(b, a^*) = dot(c, a^*) = dot(c, b^*) = 0

倒易矩阵形式：Matrix_A = array([[dot(a^*, a^*), dot(a^*, b^*), dot(a^*, c^*)],
             [dot(b^*, a^*), dot(b^*, b^*), dot(b^*, c^*)],
             [dot(c^*, a^*), dot(c^*, b^*), dot(c^*, c^*)]])

对于六方晶系，通过查表或者按照上式直接求解，得到a^*, b^*, c^*的长度分别为：2/(aqrt(3)*a),2/(aqrt(3)*b),1/c，此处a = b。
得到六方晶系的倒易矩阵为：

 Matrix_A = array([[4/3/a^2, 2/3/a^2, 0],
                   [2/3/a^2, 4/3/a^2, 0],
                [0, 0, 1/c/c]])
通过倒易矩阵，可以通过平面指数(h k l)得到法线方向指数(h1 k1 l1)。同理，可由法线方向指数(h1 k1 l1)得到平面指数(h k l)。即：

array([h1, k1, l1]) = Matrix_A * array([h1, k1, l1])

array([h1, k1, l1]) = inv(Matrix_A) * array([h1, k1, l1])
"""

import re
from math import sqrt

from numpy import array, ndarray, zeros
from numpy.linalg import norm

from pyfem.fem.constants import DTYPE
from pyfem.utils.colors import error_style

slip_dict: dict[str, tuple[ndarray, ndarray]] = {
    'fcc{111}<110>': (array([[0, 1, -1],  # B2
                             [-1, 0, 1],  # B4
                             [1, -1, 0],  # B5
                             [0, -1, -1],  # C1
                             [1, 0, 1],  # C3
                             [-1, 1, 0],  # C5
                             [0, -1, 1],  # A2
                             [-1, 0, -1],  # A3
                             [1, 1, 0],  # A6
                             [0, 1, 1],  # D1
                             [1, 0, -1],  # D4
                             [-1, -1, 0]],  # D6
                            dtype=DTYPE),
                      array([[1, 1, 1],  # B2
                             [1, 1, 1],  # B4
                             [1, 1, 1],  # B5
                             [-1, -1, 1],  # C1
                             [-1, -1, 1],  # C3
                             [-1, -1, 1],  # C5
                             [1, -1, -1],  # A2
                             [1, -1, -1],  # A3
                             [1, -1, -1],  # A6
                             [-1, 1, -1],  # D1
                             [-1, 1, -1],  # D4
                             [-1, 1, -1]],  # D6
                            dtype=DTYPE)),
    'fcc{110}<110>': (array([[1, 1, 0],
                             [1, -1, 0],
                             [1, 0, 1],
                             [1, 0, -1],
                             [0, 1, 1],
                             [0, 1, -1]],
                            dtype=DTYPE),
                      array([[1, -1, 0],
                             [1, 1, 0],
                             [1, 0, -1],
                             [1, 0, 1],
                             [0, 1, -1],
                             [0, 1, 1]],
                            dtype=DTYPE)),
    'bcc{110}<111>': (array([[1, -1, 1],  # D1
                             [-1, -1, 1],  # C1
                             [1, 1, 1],  # B2
                             [-1, 1, 1],  # A2
                             [-1, 1, 1],  # A3
                             [-1, -1, 1],  # C3
                             [1, 1, 1],  # B4
                             [1, -1, 1],  # D4
                             [-1, 1, 1],  # A6
                             [-1, 1, -1],  # D6
                             [1, 1, 1],  # B5
                             [1, 1, -1]],  # C5
                            dtype=DTYPE),
                      array([[0, 1, 1],  # D1
                             [0, 1, 1],  # C1
                             [0, -1, 1],  # B2
                             [0, -1, 1],  # A2
                             [1, 0, 1],  # A3
                             [1, 0, 1],  # C3
                             [-1, 0, 1],  # B4
                             [-1, 0, 1],  # D4
                             [1, 1, 0],  # A6
                             [1, 1, 0],  # D6
                             [-1, 1, 0],  # B5
                             [-1, 1, 0]],  # C5
                            dtype=DTYPE)),
    'bcc{112}<111>': (array([[-1, 1, 1],  # A-4
                             [1, 1, 1],  # B-3
                             [1, 1, -1],  # C-10
                             [1, -1, 1],  # D-9
                             [1, -1, 1],  # D-6
                             [1, 1, -1],  # C-5
                             [1, 1, 1],  # B-12
                             [-1, 1, 1],  # A-11
                             [1, 1, -1],  # C-2
                             [1, -1, 1],  # D-1
                             [-1, 1, 1],  # A-8
                             [1, 1, 1]],  # B-7
                            dtype=DTYPE),
                      array([[2, 1, 1],  # A-4
                             [-2, 1, 1],  # B-3
                             [2, -1, 1],  # C-10
                             [2, 1, -1],  # D-9
                             [1, 2, 1],  # D-6
                             [-1, 2, 1],  # C-5
                             [1, -2, 1],  # B-12
                             [1, 2, -1],  # A-11
                             [1, 1, 2],  # C-2
                             [-1, 1, 2],  # D-1
                             [1, -1, 2],  # A-8
                             [1, 1, -2]],  # B-7
                            dtype=DTYPE)),
    'bcc{123}<111>': (array([[1, 1, -1],
                             [1, -1, 1],
                             [-1, 1, 1],
                             [1, 1, 1],
                             [1, -1, 1],
                             [1, 1, -1],
                             [1, 1, 1],
                             [-1, 1, 1],
                             [1, 1, -1],
                             [1, -1, 1],
                             [-1, 1, 1],
                             [1, 1, 1],
                             [1, -1, 1],
                             [1, 1, -1],
                             [1, 1, 1],
                             [-1, 1, 1],
                             [-1, 1, 1],
                             [1, 1, 1],
                             [1, 1, -1],
                             [1, -1, 1],
                             [-1, 1, 1],
                             [1, 1, 1],
                             [1, 1, -1],
                             [1, -1, 1]],
                            dtype=DTYPE),
                      array([[1, 2, 3],
                             [-1, 2, 3],
                             [1, -2, 3],
                             [1, 2, -3],
                             [1, 3, 2],
                             [-1, 3, 2],
                             [1, -3, 2],
                             [1, 3, -2],
                             [2, 1, 3],
                             [-2, 1, 3],
                             [2, -1, 3],
                             [2, 1, -3],
                             [2, 3, 1],
                             [-2, 3, 1],
                             [2, -3, 1],
                             [2, 3, -1],
                             [3, 1, 2],
                             [-3, 1, 2],
                             [3, -1, 2],
                             [3, 1, -2],
                             [3, 2, 1],
                             [-3, 2, 1],
                             [3, -2, 1],
                             [3, 2, -1]],
                            dtype=DTYPE)),
    'hcp{001}<-1-10>': (array([[2, -1, -1, 0],
                               [-1, 2, -1, 0],
                               [-1, -1, 2, 0]],
                              dtype=DTYPE),
                        array([[0, 0, 0, 1],
                               [0, 0, 0, 1],
                               [0, 0, 0, 1]],
                              dtype=DTYPE)),  # basal systems (independent of c/a-ratio)
    'hcp{1-10}<-1-10>': (array([[2, -1, -1, 0],
                                [-1, 2, -1, 0],
                                [-1, -1, 2, 0]],
                               dtype=DTYPE),
                         array([[0, 1, -1, 0],
                                [-1, 0, 1, 0],
                                [1, -1, 0, 0]],
                               dtype=DTYPE)),  # prismatic systems (independent of c/a-ratio)
    'hcp{-111}<-1-10>': (array([[-1, 2, -1, 0],
                                [-2, 1, 1, 0],
                                [-1, -1, 2, 0],
                                [1, -2, 1, 0],
                                [2, -1, -1, 0],
                                [1, 1, -2, 0]],
                               dtype=DTYPE),
                         array([[1, 0, -1, 1],
                                [0, 1, -1, 1],
                                [-1, 1, 0, 1],
                                [-1, 0, 1, 1],
                                [0, -1, 1, 1],
                                [1, -1, 0, 1]],
                               dtype=DTYPE)),  # 1st order pyramidal <a> systems (direction independent of c/a-ratio)
    'hcp{-101}<113>': (array([[-2, 1, 1, 3],
                              [-1, -1, 2, 3],
                              [-1, -1, 2, 3],
                              [1, -2, 1, 3],
                              [1, -2, 1, 3],
                              [2, -1, -1, 3],
                              [2, -1, -1, 3],
                              [1, 1, -2, 3],
                              [1, 1, -2, 3],
                              [-1, 2, -1, 3],
                              [-1, 2, -1, 3],
                              [-2, 1, 1, 3]],
                             dtype=DTYPE),
                       array([[1, 0, -1, 1],
                              [1, 0, -1, 1],
                              [0, 1, -1, 1],
                              [0, 1, -1, 1],
                              [-1, 1, 0, 1],
                              [-1, 1, 0, 1],
                              [-1, 0, 1, 1],
                              [-1, 0, 1, 1],
                              [0, -1, 1, 1],
                              [0, -1, 1, 1],
                              [1, -1, 0, 1],
                              [1, -1, 0, 1]],
                             dtype=DTYPE)),  # 1st order pyramidal <c+a> systems (direction independent of c/a-ratio)
    'hcp{-112}<113>': (array([[-1, -1, 2, 3],
                              [1, -2, 1, 3],
                              [2, -1, -1, 3],
                              [1, 1, -2, 3],
                              [-1, 2, -1, 3],
                              [-2, 1, 1, 3]],
                             dtype=DTYPE),
                       array([[1, 1, -2, 2],
                              [-1, 2, -1, 2],
                              [-2, 1, 1, 2],
                              [-1, -1, 2, 2],
                              [1, -2, 1, 2],
                              [2, -1, -1, 2]],
                             dtype=DTYPE)),  # 2nd order pyramidal <c+a> systems
    # body centered tetragonal
    'bct{100}<001>': (array([[0, 0, 1],
                             [0, 0, 1]],
                            dtype=DTYPE),
                      array([[1, 0, 0],
                             [0, 1, 0]],
                            dtype=DTYPE)),
    'bct{110}<001>': (array([[0, 0, 1],
                             [0, 0, 1]],
                            dtype=DTYPE),
                      array([[1, 1, 0],
                             [-1, 1, 0]],
                            dtype=DTYPE)),
    'bct{100}<010>': (array([[0, 1, 0],
                             [1, 0, 0]],
                            dtype=DTYPE),
                      array([[1, 0, 0],
                             [0, 1, 0]],
                            dtype=DTYPE)),
    'bct{110}<1-11>': (array([[1, -1, 1],
                              [1, -1, -1],
                              [-1, -1, -1],
                              [-1, -1, 1]],
                             dtype=DTYPE),
                       array([[1, 1, 0],
                              [1, 1, 0],
                              [-1, 1, 0],
                              [-1, 1, 0]],
                             dtype=DTYPE)),
    'bct{110}<1-10>': (array([[1, -1, 0],
                              [1, 1, 0]],
                             dtype=DTYPE),
                       array([[1, 1, 0],
                              [1, -1, 0]],
                             dtype=DTYPE)),
    'bct{100}<011>': (array([[0, 1, 1],
                             [0, -1, 1],
                             [-1, 0, 1],
                             [1, 0, 1]],
                            dtype=DTYPE),
                      array([[1, 0, 0],
                             [1, 0, 0],
                             [0, 1, 0],
                             [0, 1, 0]],
                            dtype=DTYPE)),
    'bct{001}<010>': (array([[0, 1, 0],
                             [1, 0, 0]],
                            dtype=DTYPE),
                      array([[0, 0, 1],
                             [0, 0, 1]],
                            dtype=DTYPE)),
    'bct{001}<110>': (array([[1, 1, 0],
                             [-1, 1, 0]],
                            dtype=DTYPE),
                      array([[0, 0, 1],
                             [0, 0, 1]],
                            dtype=DTYPE)),
    'bct{011}<01-1>': (array([[0, 1, -1],
                              [0, -1, -1],
                              [-1, 0, -1],
                              [1, 0, -1]],
                             dtype=DTYPE),
                       array([[0, 1, 1],
                              [0, -1, 1],
                              [-1, 0, 1],
                              [1, 0, 1]],
                             dtype=DTYPE)),
    'bct{011}<1-11>': (array([[1, -1, 1],
                              [1, 1, -1],
                              [1, 1, 1],
                              [-1, 1, 1],
                              [1, -1, -1],
                              [-1, -1, 1],
                              [1, 1, 1],
                              [1, -1, 1]],
                             dtype=DTYPE),
                       array([[0, 1, 1],
                              [0, 1, 1],
                              [0, 1, -1],
                              [0, 1, -1],
                              [1, 0, 1],
                              [1, 0, 1],
                              [1, 0, -1],
                              [1, 0, -1]],
                             dtype=DTYPE)),
    'bct{011}<100>': (array([[1, 0, 0],
                             [1, 0, 0],
                             [0, 1, 0],
                             [0, 1, 0]],
                            dtype=DTYPE),
                      array([[0, 1, 1],
                             [0, 1, -1],
                             [1, 0, 1],
                             [1, 0, -1]],
                            dtype=DTYPE)),
    'bct{211}<01-1>': (array([[0, 1, -1],
                              [0, -1, -1],
                              [1, 0, -1],
                              [-1, 0, -1],
                              [0, 1, -1],
                              [0, -1, -1],
                              [-1, 0, -1],
                              [1, 0, -1]],
                             dtype=DTYPE),
                       array([[2, 1, 1],
                              [2, -1, 1],
                              [1, 2, 1],
                              [-1, 2, 1],
                              [-2, 1, 1],
                              [-2, -1, 1],
                              [-1, -2, 1],
                              [1, -2, 1]],
                             dtype=DTYPE)),
    'bct{211}<-111>': (array([[-1, 1, 1],
                              [-1, -1, 1],
                              [1, -1, 1],
                              [-1, -1, 1],
                              [1, 1, 1],
                              [1, -1, 1],
                              [-1, 1, 1],
                              [1, 1, 1]],
                             dtype=DTYPE),
                       array([[2, 1, 1],
                              [2, -1, 1],
                              [1, 2, 1],
                              [-1, 2, 1],
                              [-2, 1, 1],
                              [-2, -1, 1],
                              [-1, -2, 1],
                              [1, -2, 1]],
                             dtype=DTYPE)),
}

twin_dict: dict[str, tuple[ndarray, ndarray]] = {
    'fcc{111}<112>': (array([[1, 1, 1],
                             [1, 1, 1],
                             [1, 1, 1],
                             [-1, -1, 1],
                             [-1, -1, 1],
                             [-1, -1, 1],
                             [1, -1, -1],
                             [1, -1, -1],
                             [1, -1, -1],
                             [-1, 1, -1],
                             [-1, 1, -1],
                             [-1, 1, -1]], dtype=DTYPE),
                      array([[-2, 1, 1],
                             [1, -2, 1],
                             [1, 1, -2],
                             [2, -1, 1],
                             [-1, 2, 1],
                             [-1, -1, -2],
                             [-2, -1, -1],
                             [1, 2, -1],
                             [1, -1, 2],
                             [2, 1, -1],
                             [-1, -2, -1],
                             [-1, 1, 2]], dtype=DTYPE)),
    'bcc{112}<111>': (array([[-1, 1, 1],
                             [1, 1, 1],
                             [1, 1, -1],
                             [1, -1, 1],
                             [1, -1, 1],
                             [1, 1, -1],
                             [1, 1, 1],
                             [-1, 1, 1],
                             [1, 1, -1],
                             [1, -1, 1],
                             [-1, 1, 1],
                             [1, 1, 1]], dtype=DTYPE),
                      array([[2, 1, 1],
                             [-2, 1, 1],
                             [2, -1, 1],
                             [2, 1, -1],
                             [1, 2, 1],
                             [-1, 2, 1],
                             [1, -2, 1],
                             [1, 2, -1],
                             [1, 1, 2],
                             [-1, 1, 2],
                             [1, -1, 2],
                             [1, 1, -2]], dtype=DTYPE)),
    # hex twin systems, sorted by P. Eisenlohr CCW around <c> starting next to a_1 axis
    'hcp{102}<-101>': (array([[-1, 0, 1, 1],
                              [0, -1, 1, 1],
                              [1, -1, 0, 1],
                              [1, 0, -1, 1],
                              [0, 1, -1, 1],
                              [-1, 1, 0, 1]],
                             dtype=DTYPE),
                       array([[1, 0, -1, 2],
                              [0, 1, -1, 2],
                              [-1, 1, 0, 2],
                              [-1, 0, 1, 2],
                              [0, -1, 1, 2],
                              [1, -1, 0, 2]],
                             dtype=DTYPE)),
    # shear = (3-(c/a)^2)/(sqrt(3) c/a) tension in Co, Mg, Zr, Ti, and Be; compression in Cd and Zn
    'hcp{-1-1-1}<116>': (array([[-1, -1, 2, 6],
                                [1, -2, 1, 6],
                                [2, -1, -1, 6],
                                [1, 1, -2, 6],
                                [-1, 2, -1, 6],
                                [-2, 1, 1, 6]],
                               dtype=DTYPE),
                         array([[1, 1, -2, 1],
                                [-1, 2, -1, 1],
                                [-2, 1, 1, 1],
                                [-1, -1, 2, 1],
                                [1, -2, 1, 1],
                                [2, -1, -1, 1]],
                               dtype=DTYPE)),  # shear = 1/(c/a) tension in Co, Re, and Zr
    'hcp{101}<10-2>': (array([[1, 0, -1, -2],
                              [0, 1, -1, -2],
                              [-1, 1, 0, -2],
                              [-1, 0, 1, -2],
                              [0, -1, 1, -2],
                              [1, -1, 0, -2]],
                             dtype=DTYPE),
                       array([[1, 0, -1, 1],
                              [0, 1, -1, 1],
                              [-1, 1, 0, 1],
                              [-1, 0, 1, 1],
                              [0, -1, 1, 1],
                              [1, -1, 0, 1]],
                             dtype=DTYPE)),  # shear = (4(c/a)^2-9)/(4 sqrt(3) c/a) compression in Mg
    'hcp{112}<11-3>': (array([[1, 1, -2, -3],
                              [-1, 2, -1, -3],
                              [-2, 1, 1, -3],
                              [-1, -1, 2, -3],
                              [1, -2, 1, -3],
                              [2, -1, -1, -3]],
                             dtype=DTYPE),
                       array([[1, 1, -2, 2],
                              [-1, 2, -1, 2],
                              [-2, 1, 1, 2],
                              [-1, -1, 2, 2],
                              [1, -2, 1, 2],
                              [2, -1, -1, 2]],
                             dtype=DTYPE)),  # systems, shear = 2((c/a)^2-2)/(3 c/a) compression in Ti and Zr
}

cleavage_dict: dict[str, tuple[ndarray, ndarray]] = {
    'fcc{001}<001>': (array([[1, 0, 0],
                             [0, 1, 0],
                             [0, 0, 1]], dtype=DTYPE),
                      array([[0, 1, 0],
                             [0, 0, 1],
                             [1, 0, 0]], dtype=DTYPE)),
    'bcc{001}<001>': (array([[0, 1, 0],
                             [0, 0, 1],
                             [1, 0, 0]], dtype=DTYPE),
                      array([[1, 0, 0],
                             [0, 1, 0],
                             [0, 0, 1]], dtype=DTYPE)),
}


def process_string(string):
    result = []
    i = 0
    while i < len(string):
        if string[i] == '-':
            result.append('-' + string[i + 1])
            i += 2
        else:
            result.append(string[i])
            i += 1
    return array(result, dtype='int32')


def generate_mn(system_type: str, system_name: str, c_over_a: float) -> tuple[int, ndarray, ndarray]:
    crystal_type = system_name[0:3]

    if system_type == 'slip':
        if system_name not in slip_dict:
            raise NotImplementedError(error_style(
                f'{system_name} is not supported. The allowed slip system types are {list(slip_dict.keys())}'))
        m_0, n_0 = slip_dict[system_name]
    elif system_type == 'twin':
        if system_name not in twin_dict:
            raise NotImplementedError(error_style(
                f'{system_name} is not supported. The allowed twin system types are {list(twin_dict.keys())}'))
        m_0, n_0 = twin_dict[system_name]
    elif system_type == 'cleavage':
        if system_name not in cleavage_dict:
            raise NotImplementedError(error_style(
                f'{system_name} is not supported. The allowed cleavage system types are {list(cleavage_dict.keys())}'))
        m_0, n_0 = cleavage_dict[system_name]
    else:
        raise NotImplementedError(
            error_style(f'{system_type} is not supported. The allowed keywords are [\'slip\', \'twin\', \'cleavage\']'))

    system_number = len(m_0)

    n_pattern = r"\{(.+?)\}"
    m_pattern = r"\<(.+?)\>"
    n_vector = process_string(re.search(n_pattern, system_name).group(1))
    m_vector = process_string(re.search(m_pattern, system_name).group(1))

    if crystal_type in ['fcc', 'bcc', 'bct']:
        m = m_0 / norm(m_vector)
        n = n_0 / norm(n_vector)

    elif crystal_type in ['hcp']:
        m = zeros((system_number, 3), dtype=DTYPE)
        n = zeros((system_number, 3), dtype=DTYPE)

        m[:, 0] = m_0[:, 0] * 1.5
        m[:, 1] = (m_0[:, 0] + 2.0 * m_0[:, 1]) * sqrt(3.0) / 2.0
        m[:, 2] = m_0[:, 2] * c_over_a

        n[:, 0] = n_0[:, 0]
        n[:, 1] = (n_0[:, 0] + 2.0 * n_0[:, 1]) / sqrt(3.0)
        n[:, 2] = n_0[:, 3] / c_over_a

        m = m / norm(m_vector)
        n = n / norm(n_vector)

    else:
        raise NotImplementedError(error_style(
            f'crystal type {crystal_type} is not supported. The allowed keywords are [\'fcc\', \'bcc\', \'bct\', \'hcp\']'))

    return system_number, m, n


if __name__ == '__main__':
    system_number, m, n = generate_mn('slip', 'fcc{111}<110>', 1.0)

    system_number, m, n = generate_mn('twin', 'hcp{112}<11-3>', 1.633)

    print('m', m)
    print('n', n)
