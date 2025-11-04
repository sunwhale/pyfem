# -*- coding: utf-8 -*-
r"""
**提供晶体变形系统的信息**

本模块中包含3个字典：:py:attr:`slip_dict` ， :py:attr:`cleavage_dict` ， :py:attr:`twin_dict` 。

其中，字典 :py:attr:`slip_dict` 支持的键值为::

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

字典 :py:attr:`twin_dict` 支持的键值为::

    fcc{111}<112>
    bcc{112}<111>
    hcp{102}<-101>
    hcp{-1-1-1}<116>
    hcp{101}<10-2>
    hcp{112}<11-3>

字典 :py:attr:`cleavage_dict` 支持的键值为::

    fcc{001}<001>
    bcc{001}<001>

对于 FCC(Face Centered Cubic)、BCC(Body Centered Cubic)、BCT(Face Centered Tetragonal) 晶系，
3个字典的返回值为晶体变形系统三轴直角坐标系的晶向指数和晶面法向指数组成的元组 (np.ndarray, np.ndarray)。
根据变形系统的类型，首先通过 :py:attr:`generate_mn` 函数获得存储在字典中的晶向指数与晶面法向指数，然后再对元组中的元素归一化，
最终获得变形系统的变形数目以及单位化的晶体晶向指数和晶面法向指数。

对于 HCP(Hexagonal Close Packed) 晶系，3个字典的返回值为晶体变形系统的方向指数和平面指数组成的元组 (np.ndarray, np.ndarray)。
其方向指数和平面指数都是在非正交的四轴坐标系中用密勒-布喇菲指数表示的，但在有限元软件中一般使用笛卡尔直角坐标系，所以对于 HCP 晶系，
想要获得晶体的晶向指数与晶面法向指数，还需要进行坐标系的转换。具体流程为：第一步，根据晶体变形系统的类型，通过 :py:attr:`generate_mn` 函数获得存储在字典中的方向指数和平面指数。
第二步，将四轴坐标系的方向指数和平面指数转换为笛卡尔直角坐标系的晶向指数和晶面法向指数。第三步，对元组中的元素归一化，最终获得变形系统的变形数目以及单位化的晶体晶向指数和晶面法向指数。

以下为转化过程：

1. 将六方晶系四轴坐标系的方向指数与平面指数转化为三轴坐标系的晶向指数与晶面法向指数。

假设四轴坐标系的方向指数为 :math:`\left[ {u{\text{ }}v{\text{ }}t{\text{ }}w} \right]` ,
则三轴坐标系的晶向指数 :math:`\left[ {U{\text{ }}V{\text{ }}W} \right]` 用四轴坐标系的方向指数表示为：

.. math::
    \left( {\begin{array}{*{20}{c}}
      U \\
      V \\
      W
    \end{array}} \right) = \left[ {\begin{array}{*{20}{c}}
      2&1&0&0 \\
      1&2&0&0 \\
      0&0&0&1
    \end{array}} \right]\left( {\begin{array}{*{20}{c}}
      u \\
      v \\
      t \\
      w
    \end{array}} \right) = {A^{\left( 1 \right)}}\left( {\begin{array}{*{20}{c}}
      u \\
      v \\
      t \\
      w
    \end{array}} \right)

其中， :math:`{A^{\left( 1 \right)}}` 为四轴坐标系方向指数与三轴坐标系晶向指数的转换矩阵。

假设四轴坐标系的平面指数为 :math:`\left( {h{\text{ }}k{\text{ }}i{\text{ }}l} \right)` ，其对应的三轴坐标系的平面指数只需要去掉第三个指数，
即三轴坐标系的平面指数 :math:`\left( {{h^{\left( 1 \right)}}{\text{ }}{k^{\left( 1 \right)}}{\text{ }}{l^{\left( 1 \right)}}} \right)` 为：

.. math::
    \left( {\begin{array}{*{20}{c}}
      {{h^{\left( 1 \right)}}} \\
      {{k^{\left( 1 \right)}}} \\
      {{l^{\left( 1 \right)}}}
    \end{array}} \right) = \left[ {\begin{array}{*{20}{c}}
      1&0&0&0 \\
      0&1&0&0 \\
      0&0&0&1
    \end{array}} \right]\left( {\begin{array}{*{20}{c}}
      h \\
      k \\
      i \\
      l
    \end{array}} \right) = {B^{\left( 1 \right)}}\left( {\begin{array}{*{20}{c}}
      h \\
      k \\
      i \\
      l
    \end{array}} \right)

其中， :math:`{B^{\left( 1 \right)}}` 为四轴坐标平面指数与三轴坐标平面指数的变换矩阵。

由于六方晶系三轴坐标系的平面指数和晶面法向指数不同，根据倒易点阵的相关知识，三轴坐标系的晶面法向指数 :math:`\left( {{h^3}{\text{ }}{k^3}{\text{ }}{l^3}} \right)` 可用三轴坐标系的平面指数表示为：

.. math::
    \left( {\begin{array}{*{20}{c}}
      {{h^3}} \\
      {{k^3}} \\
      {{l^3}}
    \end{array}} \right) = T\left( {\begin{array}{*{20}{c}}
      {{h^{\left( 1 \right)}}} \\
      {{k^{\left( 1 \right)}}} \\
      {{l^{\left( 1 \right)}}}
    \end{array}} \right) = \left[ {\begin{array}{*{20}{c}}
      {\frac{4}{3}}&{\frac{2}{3}}&0 \\
      {\frac{2}{3}}&{\frac{4}{3}}&0 \\
      0&0&{{{\left( {\frac{a}{c}} \right)}^2}}
    \end{array}} \right]{B^{\left( 1 \right)}}\left( {\begin{array}{*{20}{c}}
      h \\
      k \\
      i \\
      l
    \end{array}} \right) = \left[ {\begin{array}{*{20}{c}}
      {\frac{4}{3}}&{\frac{2}{3}}&0&0 \\
      {\frac{2}{3}}&{\frac{4}{3}}&0&0 \\
      0&0&0&{{{\left( {\frac{a}{c}} \right)}^2}}
    \end{array}} \right]\left( {\begin{array}{*{20}{c}}
      h \\
      k \\
      i \\
      l
    \end{array}} \right) = {B^{\left( 2 \right)}}\left( {\begin{array}{*{20}{c}}
      h \\
      k \\
      i \\
      l
    \end{array}} \right)

其中， :math:`c/a` 为六方晶系的晶轴比， :math:`T` 为六方晶系三轴坐标系平面指数与三轴坐标系晶面法向指数的转换矩阵，
由倒易点阵性质求得，见相关知识的补充。 :math:`{B^{\left( 2 \right)}}` 为六方晶系四轴坐标系平面指数与三轴坐标系晶面法向指数的变换矩阵。

2. 将三轴坐标系的晶向指数 :math:`\left[ {U{\text{ }}V{\text{ }}W} \right]` 和晶面法向指数 :math:`\left( {{h^3}{\text{ }}{k^3}{\text{ }}{l^3}} \right)` 转换为直角坐标系的晶向指数与晶面法向指数。

六方晶系三轴坐标系和直角坐标系的相对位置如下图所示::

                           z, c
                           *
                           *
                           *
                           *
                           *
                           *
                           *
                           *
                           *
                           *
                           *
                           *
                           *
                           *
                           o * * * * * * * * * * * * * * * * y, a2
                         **   |
                       * *---90°
                     *  *
                   *   *
                 *    *
               *     *
             *      *
           *\      *
         *   30°--*
       *         *
    a1          *
               *
              x


其中，六方晶系的三轴坐标系是将单胞的 :math:`{\mathbf{a}}` 和 :math:`{\mathbf{b}}` 轴标为 :math:`{{{\mathbf{a}}_1}}` 和 :math:`{{{\mathbf{a}}_2}}` 轴， :math:`{\mathbf{c}}` 轴保持不变。 :math:`{\mathbf{c}}` 轴
垂直于 :math:`{{{\mathbf{a}}_1}} - {{{\mathbf{a}}_2}}` 平面， :math:`{{{\mathbf{a}}_1}} - {{{\mathbf{a}}_2}}` 轴的夹角为 :math:`{120^ \circ }` 。 :math:`{{{\mathbf{a}}_1}} - {{{\mathbf{a}}_2}}` 轴基矢
的模： :math:`\left| {{{\mathbf{a}}_1}} \right| = \left| {{{\mathbf{a}}_2}} \right| = a` ， :math:`{\mathbf{c}}` 轴基矢的模： :math:`\left| {\mathbf{c}} \right|  = c` 。

三轴坐标系的 :math:`{{{\mathbf{a}}_2}}` 轴与直角坐标系的 :math:`{\mathbf{y}}` 轴平行，三轴坐标系的  :math:`{\mathbf{c}}` 轴与直角坐标系的  :math:`{\mathbf{z}}` 轴平行， :math:`{\mathbf{x}}` 轴与 :math:`{{{\mathbf{a}}_1}}` 轴之间的夹角为 :math:`{30^ \circ }` 。
得到转换公式：

.. math::
    \left( {\begin{array}{*{20}{c}}
      x \\
      y \\
      z
    \end{array}} \right) = \left[ {\begin{array}{*{20}{c}}
      {\cos ({{30}^ \circ })}&0&0 \\
      { - \sin ({{30}^ \circ })}&1&0 \\
      0&0&{\frac{c}{a}}
    \end{array}} \right]\left( {\begin{array}{*{20}{c}}
      {{a_1}} \\
      {{a_2}} \\
      c
    \end{array}} \right) = \left[ {\begin{array}{*{20}{c}}
      {\frac{{\sqrt 3 }}{2}}&0&0 \\
      { - \frac{1}{2}}&1&0 \\
      0&0&{\frac{c}{a}}
    \end{array}} \right]\left( {\begin{array}{*{20}{c}}
      {{a_1}} \\
      {{a_2}} \\
      c
    \end{array}} \right) = C\left( {\begin{array}{*{20}{c}}
      {{a_1}} \\
      {{a_2}} \\
      c
    \end{array}} \right)

其中，矩阵 :math:`C` 是三轴坐标系和直角坐标系的变换矩阵。

利用变换矩阵 :math:`C` 并结合之前计算得到的三轴坐标系晶向指数 :math:`\left[ {U{\text{ }}V{\text{ }}W} \right]` 和晶面法向指数 :math:`\left( {{h^3}{\text{ }}{k^3}{\text{ }}{l^3}} \right)` ，
可计算得出直角坐标系的晶向指数 :math:`\left[ {{w^r}{\text{ }}{v^r}{\text{ }}{w^r}} \right]` 为：

.. math::
    \left( {\begin{array}{*{20}{c}}
      {{u^r}} \\
      {{v^r}} \\
      {{w^r}}
    \end{array}} \right) = C\left( {\begin{array}{*{20}{c}}
      U \\
      V \\
      W
    \end{array}} \right) = C{A^{\left( 1 \right)}}\left( {\begin{array}{*{20}{c}}
      u \\
      v \\
      t \\
      w
    \end{array}} \right) = \left[ {\begin{array}{*{20}{c}}
      {\sqrt 3 }&{\frac{{\sqrt 3 }}{2}}&0&0 \\
      0&{\frac{3}{2}}&0&0 \\
      0&0&0&{\frac{c}{a}}
    \end{array}} \right]\left( {\begin{array}{*{20}{c}}
      u \\
      v \\
      t \\
      w
    \end{array}} \right) = \left( {\begin{array}{*{20}{c}}
      {\frac{{\sqrt 3 }}{2}\left( {2u + v} \right)} \\
      {\frac{3}{2}v} \\
      {\frac{c}{a}w}
    \end{array}} \right)

直角坐标系的晶面法向指数 :math:`\left( {{h^r}{\text{ }}{k^r}{\text{ }}{l^r}} \right)` 为：

.. math::
    \left( {\begin{array}{*{20}{c}}
      {{h^r}} \\
      {{k^r}} \\
      {{l^r}}
    \end{array}} \right) = C\left( {\begin{array}{*{20}{c}}
      {{h^3}} \\
      {{k^3}} \\
      {{l^3}}
    \end{array}} \right) = C{B^{\left( 2 \right)}}\left( {\begin{array}{*{20}{c}}
      h \\
      k \\
      i \\
      l
    \end{array}} \right) = \left[ {\begin{array}{*{20}{c}}
      {\frac{2}{{\sqrt 3 }}}&{\frac{1}{{\sqrt 3 }}}&0&0 \\
      0&1&0&0 \\
      0&0&0&{\frac{a}{c}}
    \end{array}} \right]\left( {\begin{array}{*{20}{c}}
      h \\
      k \\
      i \\
      l
    \end{array}} \right) = \left( {\begin{array}{*{20}{c}}
      {\frac{1}{{\sqrt 3 }}\left( {2h + k} \right)} \\
      k \\
      0 \\
      {\frac{a}{c}l}
    \end{array}} \right)

最后对直角坐标系的晶向指数 :math:`\left[ {{w^r}{\text{ }}{v^r}{\text{ }}{w^r}} \right]` 和晶面法向指数 :math:`\left( {{h^r}{\text{ }}{k^r}{\text{ }}{l^r}} \right)` 归一化，
得到六方晶系直角坐标系下单位化的晶向指数和晶面法向指数，即可应用于后续计算。

**倒易矩阵相关知识补充:**

1. 定义

有两种点阵，它们的点阵参数分别为 :math:`{\mathbf{a}},{\mathbf{b}},{\mathbf{c}},\alpha ,\beta ,\gamma` 和 :math:`{{\mathbf{a}}^*},{{\mathbf{b}}^*},{{\mathbf{c}}^*},{\alpha ^*},{\beta ^*},{\gamma ^*}` 。
用符号 :math:`\left( {\square ,\square } \right)` 表示两个矢量的内积。两种点阵的点阵参数之间存在以下关系：

.. math::
    \left( {{\mathbf{a}},{{\mathbf{a}}^*}} \right) = \left( {{\mathbf{b}},{{\mathbf{b}}^*}} \right) = \left( {{\mathbf{c}},{{\mathbf{c}}^*}} \right) = 1

.. math::
    \left( {{\mathbf{a}},{{\mathbf{b}}^*}} \right) = \left( {{\mathbf{a}},{{\mathbf{c}}^*}} \right) = \left( {{\mathbf{b}},{{\mathbf{c}}^*}} \right) = \left( {{\mathbf{b}},{{\mathbf{a}}^*}} \right) = \left( {{\mathbf{c}}, {{\mathbf{a}}^*}} \right) = \left( {{\mathbf{c}},{{\mathbf{b}}^*}} \right) = 0

则这两个点阵互为倒易。如果 :math:`{\mathbf{a}},{\mathbf{b}},{\mathbf{c}},\alpha ,\beta ,\gamma` 确定的点阵是真实点阵（正点阵）的点阵参数,
则 :math:`{{\mathbf{a}}^*},{{\mathbf{b}}^*},{{\mathbf{c}}^*} ,{\alpha ^*},{\beta ^*},{\gamma ^*}` 确定的点阵是前者的倒易点阵。根据定义， :math:`{{\mathbf{a}}^*},{{\mathbf{b}}^*},{{\mathbf{c}}^*}` 分别垂直于 :math:`bc,ca,ab` 平面。

2. 倒易矢量在晶体学几何关系中的应用

2.1 求点阵平面的法线方向指数

对于六方晶系，由于正点阵中的面 :math:`\left( {{h}{\text{ }}{k}{\text{ }}{l}} \right)` 与其晶面法向指数 :math:`\left( {{h^*}{\text{ }}{k^*}{\text{ }}{l^*}} \right)` 一般不同名，
但是， :math:`\left( {{h}{\text{ }}{k}{\text{ }}{l}} \right)` 一定和与它同名的倒易矢量 :math:`{\left( {h{\text{ }}k{\text{ }}l} \right)^*}` 垂直，即 :math:`\left( {{h^*}{\text{ }}{k^*}{\text{ }}{l^*}} \right)\parallel {\left( {h{\text{ }}k{\text{ }}l} \right)^*}` 。
当只考虑方向，不考虑矢量的绝对长度，有

.. math::
    h{{\mathbf{a}}^*} + k{{\mathbf{b}}^*} + l{{\mathbf{c}}^*} = {h^*}{\mathbf{a}} + {k^*}{\mathbf{b}} + {l^*}{\mathbf{c}}

用 :math:`{{\mathbf{a}}^*},{{\mathbf{b}}^*},{{\mathbf{c}}^*}` 同时点乘上式两端，根据点阵参数之间的性质，得到：

.. math::
    \begin{gathered}
      {h^*} = h({{\mathbf{a}}^*},{{\mathbf{a}}^*}) + k({{\mathbf{a}}^*},{{\mathbf{b}}^*}) + l({{\mathbf{a}}^*},{{\mathbf{c}}^*}) \hfill \\
      {k^*} = h({{\mathbf{b}}^*},{{\mathbf{a}}^*}) + k({{\mathbf{b}}^*},{{\mathbf{b}}^*}) + l({{\mathbf{b}}^*},{{\mathbf{c}}^*}) \hfill \\
      {l^*} = h({{\mathbf{c}}^*},{{\mathbf{a}}^*}) + k({{\mathbf{c}}^*},{{\mathbf{b}}^*}) + l({{\mathbf{c}}^*},{{\mathbf{c}}^*}) \hfill \\
    \end{gathered}

把上面三个式子写成矩阵形式：

.. math::
    \left( {\begin{array}{*{20}{l}}
      {{h^*}} \\
      {{k^*}} \\
      {{l^*}}
    \end{array}} \right) = \left[ {\begin{array}{*{20}{c}}
      {({{\mathbf{a}}^*},{{\mathbf{a}}^*})}&{({{\mathbf{a}}^*},{{\mathbf{b}}^*})}&{({{\mathbf{a}}^*},{{\mathbf{c}}^*})} \\
      {({{\mathbf{b}}^*},{{\mathbf{a}}^*})}&{({{\mathbf{b}}^*},{{\mathbf{b}}^*})}&{({{\mathbf{b}}^*},{{\mathbf{c}}^*})} \\
      {({{\mathbf{c}}^*},{{\mathbf{a}}^*})}&{({{\mathbf{c}}^*},{{\mathbf{b}}^*})}&{({{\mathbf{c}}^*},{{\mathbf{c}}^*})}
    \end{array}} \right]\left( {\begin{array}{*{20}{l}}
      h \\
      k \\
      l
    \end{array}} \right) = D\left( {\begin{array}{*{20}{l}}
      h \\
      k \\
      l
    \end{array}} \right)

其中， :math:`D` 是平面指数和晶面法向指数的变换矩阵。当知道倒易点阵基矢 :math:`{{\mathbf{a}}^*},{{\mathbf{b}}^*},{{\mathbf{c}}^*}` 后就可求平面 :math:`\left( {{h}{\text{ }}{k}{\text{ }}{l}} \right)` 的
法线方向指数(晶面法向指数) :math:`\left( {{h^*}{\text{ }}{k^*}{\text{ }}{l^*}} \right)` 。

同理，可由晶面法线方向指数 :math:`\left( {{h^*}{\text{ }}{k^*}{\text{ }}{l^*}} \right)` 得到平面指数 :math:`\left( {{h}{\text{ }}{k}{\text{ }}{l}} \right)` 。即:

.. math::
    \left( {\begin{array}{*{20}{l}}
      h \\
      k \\
      l
    \end{array}} \right) = \left[ {\begin{array}{*{20}{c}}
      {({\mathbf{a}},{\mathbf{a}})}&{({\mathbf{a}},{\mathbf{b}})}&{({\mathbf{a}},{\mathbf{c}})} \\
      {({\mathbf{b}},{\mathbf{a}})}&{({\mathbf{b}},{\mathbf{b}})}&{({\mathbf{b}},{\mathbf{c}})} \\
      {({\mathbf{c}},{\mathbf{a}})}&{({\mathbf{c}},{\mathbf{b}})}&{({\mathbf{c}},{\mathbf{c}})}
    \end{array}} \right]\left( {\begin{array}{*{20}{l}}
      {{h^*}} \\
      {{k^*}} \\
      {{l^*}}
    \end{array}} \right) = E\left( {\begin{array}{*{20}{l}}
      {{h^*}} \\
      {{k^*}} \\
      {{l^*}}
    \end{array}} \right) = {D^{ - 1}}\left( {\begin{array}{*{20}{l}}
      {{h^*}} \\
      {{k^*}} \\
      {{l^*}}
    \end{array}} \right)

对于六方晶系，通过查表或者按照定义直接求解，得到 :math:`{{\mathbf{a}}^*},{{\mathbf{b}}^*},{{\mathbf{c}}^*}` 的长度分别为 :math:`\frac{2}{{\sqrt 3 }}a,\frac{2}{{\sqrt 3 }}a,\frac{1}{c}` ，此处 :math:`a,c` 分别对应三轴坐标系 :math:`{{{\mathbf{a}}_1}},{{{\mathbf{a}}_2}}` 和 :math:`{\mathbf{c}}` 轴的模长。
代入平面指数与晶面法向指数变换矩阵 :math:`D` ，得到：

.. math::
    {D_{{\text{HCP}}}} = \left[ {\begin{array}{*{20}{c}}
      {\frac{4}{3}{a^2}}&{\frac{2}{3}{a^2}}&0 \\
      {\frac{2}{3}{a^2}}&{\frac{4}{3}{a^2}}&0 \\
      0&0&{\frac{1}{{{c^2}}}}
    \end{array}} \right]

将 :math:`D_{{\text{HCP}}}` 乘以 :math:`{{a}^2}` ，即得到与三轴坐标系晶面法向指数和三轴坐标系平面指数转换矩阵 :math:`T` 相同的形式。

**参考书：材料科学基础-第2版-余永宁，P35-48**

"""

from math import sqrt

import numpy as np

from pyfem.fem.constants import DTYPE
from pyfem.utils.colors import error_style

slip_dict: dict[str, tuple[np.ndarray, np.ndarray]] = {
    'fcc{111}<110>': (np.array([[0, 1, -1],  # B2
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
                      np.array([[1, 1, 1],  # B2
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
    'fcc{110}<110>': (np.array([[1, 1, 0],
                                [1, -1, 0],
                                [1, 0, 1],
                                [1, 0, -1],
                                [0, 1, 1],
                                [0, 1, -1]],
                               dtype=DTYPE),
                      np.array([[1, -1, 0],
                                [1, 1, 0],
                                [1, 0, -1],
                                [1, 0, 1],
                                [0, 1, -1],
                                [0, 1, 1]],
                               dtype=DTYPE)),
    'bcc{110}<111>': (np.array([[1, -1, 1],  # D1
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
                      np.array([[0, 1, 1],  # D1
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
    'bcc{112}<111>': (np.array([[-1, 1, 1],  # A-4
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
                      np.array([[2, 1, 1],  # A-4
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
    'bcc{123}<111>': (np.array([[1, 1, -1],
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
                      np.array([[1, 2, 3],
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
    'hcp{001}<-1-10>': (np.array([[2, -1, -1, 0],
                                  [-1, 2, -1, 0],
                                  [-1, -1, 2, 0]],
                                 dtype=DTYPE),
                        np.array([[0, 0, 0, 1],
                                  [0, 0, 0, 1],
                                  [0, 0, 0, 1]],
                                 dtype=DTYPE)),  # basal systems (independent of c/a-ratio)
    'hcp{1-10}<-1-10>': (np.array([[2, -1, -1, 0],
                                   [-1, 2, -1, 0],
                                   [-1, -1, 2, 0]],
                                  dtype=DTYPE),
                         np.array([[0, 1, -1, 0],
                                   [-1, 0, 1, 0],
                                   [1, -1, 0, 0]],
                                  dtype=DTYPE)),  # prismatic systems (independent of c/a-ratio)
    'hcp{-111}<-1-10>': (np.array([[-1, 2, -1, 0],
                                   [-2, 1, 1, 0],
                                   [-1, -1, 2, 0],
                                   [1, -2, 1, 0],
                                   [2, -1, -1, 0],
                                   [1, 1, -2, 0]],
                                  dtype=DTYPE),
                         np.array([[1, 0, -1, 1],
                                   [0, 1, -1, 1],
                                   [-1, 1, 0, 1],
                                   [-1, 0, 1, 1],
                                   [0, -1, 1, 1],
                                   [1, -1, 0, 1]],
                                  dtype=DTYPE)),  # 1st order pyramidal <a> systems (direction independent of c/a-ratio)
    'hcp{-101}<113>': (np.array([[-2, 1, 1, 3],
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
                       np.array([[1, 0, -1, 1],
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
    'hcp{-112}<113>': (np.array([[-1, -1, 2, 3],
                                 [1, -2, 1, 3],
                                 [2, -1, -1, 3],
                                 [1, 1, -2, 3],
                                 [-1, 2, -1, 3],
                                 [-2, 1, 1, 3]],
                                dtype=DTYPE),
                       np.array([[1, 1, -2, 2],
                                 [-1, 2, -1, 2],
                                 [-2, 1, 1, 2],
                                 [-1, -1, 2, 2],
                                 [1, -2, 1, 2],
                                 [2, -1, -1, 2]],
                                dtype=DTYPE)),  # 2nd order pyramidal <c+a> systems
    # body centered tetragonal
    'bct{100}<001>': (np.array([[0, 0, 1],
                                [0, 0, 1]],
                               dtype=DTYPE),
                      np.array([[1, 0, 0],
                                [0, 1, 0]],
                               dtype=DTYPE)),
    'bct{110}<001>': (np.array([[0, 0, 1],
                                [0, 0, 1]],
                               dtype=DTYPE),
                      np.array([[1, 1, 0],
                                [-1, 1, 0]],
                               dtype=DTYPE)),
    'bct{100}<010>': (np.array([[0, 1, 0],
                                [1, 0, 0]],
                               dtype=DTYPE),
                      np.array([[1, 0, 0],
                                [0, 1, 0]],
                               dtype=DTYPE)),
    'bct{110}<1-11>': (np.array([[1, -1, 1],
                                 [1, -1, -1],
                                 [-1, -1, -1],
                                 [-1, -1, 1]],
                                dtype=DTYPE),
                       np.array([[1, 1, 0],
                                 [1, 1, 0],
                                 [-1, 1, 0],
                                 [-1, 1, 0]],
                                dtype=DTYPE)),
    'bct{110}<1-10>': (np.array([[1, -1, 0],
                                 [1, 1, 0]],
                                dtype=DTYPE),
                       np.array([[1, 1, 0],
                                 [1, -1, 0]],
                                dtype=DTYPE)),
    'bct{100}<011>': (np.array([[0, 1, 1],
                                [0, -1, 1],
                                [-1, 0, 1],
                                [1, 0, 1]],
                               dtype=DTYPE),
                      np.array([[1, 0, 0],
                                [1, 0, 0],
                                [0, 1, 0],
                                [0, 1, 0]],
                               dtype=DTYPE)),
    'bct{001}<010>': (np.array([[0, 1, 0],
                                [1, 0, 0]],
                               dtype=DTYPE),
                      np.array([[0, 0, 1],
                                [0, 0, 1]],
                               dtype=DTYPE)),
    'bct{001}<110>': (np.array([[1, 1, 0],
                                [-1, 1, 0]],
                               dtype=DTYPE),
                      np.array([[0, 0, 1],
                                [0, 0, 1]],
                               dtype=DTYPE)),
    'bct{011}<01-1>': (np.array([[0, 1, -1],
                                 [0, -1, -1],
                                 [-1, 0, -1],
                                 [1, 0, -1]],
                                dtype=DTYPE),
                       np.array([[0, 1, 1],
                                 [0, -1, 1],
                                 [-1, 0, 1],
                                 [1, 0, 1]],
                                dtype=DTYPE)),
    'bct{011}<1-11>': (np.array([[1, -1, 1],
                                 [1, 1, -1],
                                 [1, 1, 1],
                                 [-1, 1, 1],
                                 [1, -1, -1],
                                 [-1, -1, 1],
                                 [1, 1, 1],
                                 [1, -1, 1]],
                                dtype=DTYPE),
                       np.array([[0, 1, 1],
                                 [0, 1, 1],
                                 [0, 1, -1],
                                 [0, 1, -1],
                                 [1, 0, 1],
                                 [1, 0, 1],
                                 [1, 0, -1],
                                 [1, 0, -1]],
                                dtype=DTYPE)),
    'bct{011}<100>': (np.array([[1, 0, 0],
                                [1, 0, 0],
                                [0, 1, 0],
                                [0, 1, 0]],
                               dtype=DTYPE),
                      np.array([[0, 1, 1],
                                [0, 1, -1],
                                [1, 0, 1],
                                [1, 0, -1]],
                               dtype=DTYPE)),
    'bct{211}<01-1>': (np.array([[0, 1, -1],
                                 [0, -1, -1],
                                 [1, 0, -1],
                                 [-1, 0, -1],
                                 [0, 1, -1],
                                 [0, -1, -1],
                                 [-1, 0, -1],
                                 [1, 0, -1]],
                                dtype=DTYPE),
                       np.array([[2, 1, 1],
                                 [2, -1, 1],
                                 [1, 2, 1],
                                 [-1, 2, 1],
                                 [-2, 1, 1],
                                 [-2, -1, 1],
                                 [-1, -2, 1],
                                 [1, -2, 1]],
                                dtype=DTYPE)),
    'bct{211}<-111>': (np.array([[-1, 1, 1],
                                 [-1, -1, 1],
                                 [1, -1, 1],
                                 [-1, -1, 1],
                                 [1, 1, 1],
                                 [1, -1, 1],
                                 [-1, 1, 1],
                                 [1, 1, 1]],
                                dtype=DTYPE),
                       np.array([[2, 1, 1],
                                 [2, -1, 1],
                                 [1, 2, 1],
                                 [-1, 2, 1],
                                 [-2, 1, 1],
                                 [-2, -1, 1],
                                 [-1, -2, 1],
                                 [1, -2, 1]],
                                dtype=DTYPE)),
}

twin_dict: dict[str, tuple[np.ndarray, np.ndarray]] = {
    'fcc{111}<112>': (np.array([[1, 1, 1],
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
                      np.array([[-2, 1, 1],
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
    'bcc{112}<111>': (np.array([[-1, 1, 1],
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
                      np.array([[2, 1, 1],
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
    'hcp{102}<-101>': (np.array([[-1, 0, 1, 1],
                                 [0, -1, 1, 1],
                                 [1, -1, 0, 1],
                                 [1, 0, -1, 1],
                                 [0, 1, -1, 1],
                                 [-1, 1, 0, 1]],
                                dtype=DTYPE),
                       np.array([[1, 0, -1, 2],
                                 [0, 1, -1, 2],
                                 [-1, 1, 0, 2],
                                 [-1, 0, 1, 2],
                                 [0, -1, 1, 2],
                                 [1, -1, 0, 2]],
                                dtype=DTYPE)),
    # shear = (3-(c/a)^2)/(sqrt(3) c/a) tension in Co, Mg, Zr, Ti, and Be; compression in Cd and Zn
    'hcp{-1-1-1}<116>': (np.array([[-1, -1, 2, 6],
                                   [1, -2, 1, 6],
                                   [2, -1, -1, 6],
                                   [1, 1, -2, 6],
                                   [-1, 2, -1, 6],
                                   [-2, 1, 1, 6]],
                                  dtype=DTYPE),
                         np.array([[1, 1, -2, 1],
                                   [-1, 2, -1, 1],
                                   [-2, 1, 1, 1],
                                   [-1, -1, 2, 1],
                                   [1, -2, 1, 1],
                                   [2, -1, -1, 1]],
                                  dtype=DTYPE)),  # shear = 1/(c/a) tension in Co, Re, and Zr
    'hcp{101}<10-2>': (np.array([[1, 0, -1, -2],
                                 [0, 1, -1, -2],
                                 [-1, 1, 0, -2],
                                 [-1, 0, 1, -2],
                                 [0, -1, 1, -2],
                                 [1, -1, 0, -2]],
                                dtype=DTYPE),
                       np.array([[1, 0, -1, 1],
                                 [0, 1, -1, 1],
                                 [-1, 1, 0, 1],
                                 [-1, 0, 1, 1],
                                 [0, -1, 1, 1],
                                 [1, -1, 0, 1]],
                                dtype=DTYPE)),  # shear = (4(c/a)^2-9)/(4 sqrt(3) c/a) compression in Mg
    'hcp{112}<11-3>': (np.array([[1, 1, -2, -3],
                                 [-1, 2, -1, -3],
                                 [-2, 1, 1, -3],
                                 [-1, -1, 2, -3],
                                 [1, -2, 1, -3],
                                 [2, -1, -1, -3]],
                                dtype=DTYPE),
                       np.array([[1, 1, -2, 2],
                                 [-1, 2, -1, 2],
                                 [-2, 1, 1, 2],
                                 [-1, -1, 2, 2],
                                 [1, -2, 1, 2],
                                 [2, -1, -1, 2]],
                                dtype=DTYPE)),  # systems, shear = 2((c/a)^2-2)/(3 c/a) compression in Ti and Zr
}

cleavage_dict: dict[str, tuple[np.ndarray, np.ndarray]] = {
    'fcc{001}<001>': (np.array([[1, 0, 0],
                                [0, 1, 0],
                                [0, 0, 1]], dtype=DTYPE),
                      np.array([[0, 1, 0],
                                [0, 0, 1],
                                [1, 0, 0]], dtype=DTYPE)),
    'bcc{001}<001>': (np.array([[0, 1, 0],
                                [0, 0, 1],
                                [1, 0, 0]], dtype=DTYPE),
                      np.array([[1, 0, 0],
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
    return np.array(result, dtype='int32')


def generate_mn(system_type: str, system_name: str, c_over_a: float) -> tuple[int, np.ndarray, np.ndarray]:
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

    # n_pattern = r"\{(.+?)\}"
    # m_pattern = r"\<(.+?)\>"
    # n_vector = process_string(re.search(n_pattern, system_name).group(1))
    # m_vector = process_string(re.search(m_pattern, system_name).group(1))

    if crystal_type in ['fcc', 'bcc', 'bct']:
        m = m_0 / np.linalg.norm(m_0, axis=1).reshape((system_number, 1))
        n = n_0 / np.linalg.norm(n_0, axis=1).reshape((system_number, 1))

    elif crystal_type in ['hcp']:
        m = np.zeros((system_number, 3), dtype=DTYPE)
        n = np.zeros((system_number, 3), dtype=DTYPE)

        # m[:, 0] = m_0[:, 0] * 1.5
        # m[:, 1] = (m_0[:, 0] + 2.0 * m_0[:, 1]) * sqrt(3.0) / 2.0
        # m[:, 2] = m_0[:, 2] * c_over_a

        # n[:, 0] = n_0[:, 0]
        # n[:, 1] = (n_0[:, 0] + 2.0 * n_0[:, 1]) / sqrt(3.0)
        # n[:, 2] = n_0[:, 3] / c_over_a

        A_m = np.array([[sqrt(3.0), sqrt(3.0) / 2.0, 0, 0],
                        [0.0, 1.5, 0, 0],
                        [0, 0, 0, c_over_a]])

        A_n = np.array([[2.0 / sqrt(3.0), sqrt(3.0), 0, 0],
                        [0.0, 1.0, 0, 0],
                        [0, 0, 0, 1.0 / c_over_a]])

        m = np.dot(m_0, np.transpose(A_m))
        n = np.dot(n_0, np.transpose(A_n))

        m = m / np.linalg.norm(m, axis=1).reshape((system_number, 1))
        n = n / np.linalg.norm(n, axis=1).reshape((system_number, 1))

    else:
        raise NotImplementedError(error_style(
            f'crystal type {crystal_type} is not supported. The allowed keywords are [\'fcc\', \'bcc\', \'bct\', \'hcp\']'))

    return system_number, m, n


if __name__ == '__main__':
    system_number, m, n = generate_mn('slip', 'fcc{111}<110>', 1.0)
    system_number, m, n = generate_mn('twin', 'hcp{112}<11-3>', 1.633)

    print('m', m)
    print('n', n)
