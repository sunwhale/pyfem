位移协调元的最小势能原理
================================================================================

根据式\ :math:numref:`eq:Pi_1`\ ，弹性体最小势能原理的泛函可以表示为

.. math::
   \begin{equation}
   \Pi  = \iiint\limits_V {\left[ {A\left( {{\varepsilon _{ij}}} \right){\text{ - }}{f_i}{u_i}} \right]{\text{d}}V} - \iint\limits_{{S_p}} {{{\bar p}_i}{u_i}{\text{d}}S}
   \end{equation}

其中应变和位移满足几何方程，位移边界条件为变分约束条件，物理方程为非变分约束条件。
首先对求解域进行离散，把求解域\ :math:`V`\ 分割为\ :math:`N`\ 个有限单元，其中\ :math:`m`\ 号有限单元子域为\ :math:`{{V^{\left( m \right)}}}`\ ，外表面为\ :math:`{{S^{\left( m \right)}}}`\ 。设在相邻有限元之间的交界面上位移函数\ :math:`u_{i}`
是连续的，这种有限元称为位移协调元。位移协调元要求单元位移函数满足下列条件：

（1）在每个单元中是连续的和单值的；

（2）在单元的交界面上是协调的，即\ :math:`u_{i}^{(m)}=u_{i}^{\left(m^{\prime}\right)} \quad\left(\text { 在 } S^{\left(m m^{\prime}\right)} \text { 上 }\right)`\ ；

（3）所有含有 :math:`S_{u}`
的单元，都要满足位移边界条件\ :math:`u_{i}=\bar{u}_{i} \quad\left(\text { 在 } S_{u} \text { 上 }\right)`

如果位移函数的选择满足以上三个条件，则基于位移协调元的最小势能原理的泛函可以改写成：

.. math::
   \begin{equation}
   {\Pi ^*} = \sum\limits_{m = 1}^N {\left\{ {\iiint\limits_{{V^{\left( m \right)}}} {\left[ {{A^{\left( m \right)}}\left( {{\varepsilon _{ij}}} \right) - {f_i}u_i^{\left( m \right)}} \right]{\text{d}}V} - \iint\limits_{S_p^{\left( m \right)}} {{{\bar p}_i}u_i^{\left( m \right)}{\text{d}}S}} \right\}}
   \label{eq:Pi*_1}
   \end{equation}
   :label: eq:Pi*_1

对上式求一阶变分, 得

.. math::
   \begin{equation}
   \delta {\Pi ^*} = \sum\limits_{m = 1}^N {\left\{ {\iiint\limits_{{V^{\left( m \right)}}} {\left[ {\frac{{\partial {A^{\left( m \right)}}}}{{\partial \varepsilon _{ij}^{\left( m \right)}}}\delta \varepsilon _{ij}^{\left( m \right)} - {f_i}\delta u_i^{\left( m \right)}} \right]{\text{d}}V} - \iint\limits_{S_p^{\left( m \right)}} {{{\bar p}_i}\delta u_i^{\left( m \right)}{\text{d}}S}} \right\}}
   \label{eq:delta_Pi*_1}
   \end{equation}
   :label: eq:delta_Pi*_1

对于第\ :math:`m`\ 个单元有

.. math::
   \begin{equation}
   \frac{{\partial {A^{\left( m \right)}}}}{{\partial \varepsilon _{ij}^{\left( m \right)}}}\delta \varepsilon _{ij}^{\left( m \right)} = \frac{{\partial {A^{\left( m \right)}}}}{{\partial \varepsilon _{ij}^{\left( m \right)}}}\delta u_{i,j}^{\left( m \right)}
   \end{equation}

.. math::
   \begin{equation}
   {\left( {\frac{{\partial {A^{\left( m \right)}}}}{{\partial \varepsilon _{ij}^{\left( m \right)}}}\delta u_i^{\left( m \right)}} \right)_{,j}} = \frac{{\partial {A^{\left( m \right)}}}}{{\partial \varepsilon _{ij}^{\left( m \right)}}}\delta u_{i,j}^{\left( m \right)} + {\left( {\frac{{\partial {A^{\left( m \right)}}}}{{\partial \varepsilon _{ij}^{\left( m \right)}}}} \right)_{,j}}\delta u_i^{\left( m \right)}
   \end{equation}

第\ :math:`m`\ 个单元上应用高斯散度定理得

.. math::
   \begin{equation}
   \iiint\limits_V {{{\left( {\frac{{\partial {A^{\left( m \right)}}}}{{\partial \varepsilon _{ij}^{\left( m \right)}}}\delta u_i^{\left( m \right)}} \right)}_{,j}}{\text{d}}V} = \iint\limits_{{S^{\left( m \right)}}} {\frac{{\partial {A^{\left( m \right)}}}}{{\partial \varepsilon _{ij}^{\left( m \right)}}}n_j^{\left( m \right)}\delta u_i^{\left( m \right)}{\text{d}}S}
   \end{equation}

因此式\ :math:numref:`eq:delta_Pi*_1`\ 右边第一项

.. math::
   \begin{equation}
   \begin{array}{*{20}{l}}
     {\iiint\limits_{{V^{\left( m \right)}}} {\frac{{\partial {A^{\left( m \right)}}}}{{\partial \varepsilon _{ij}^{\left( m \right)}}}\delta \varepsilon _{ij}^{\left( m \right)}{\text{d}}V}}&{ = \iiint\limits_{{V^{\left( m \right)}}} {\frac{{\partial {A^{\left( m \right)}}}}{{\partial \varepsilon _{ij}^{\left( m \right)}}}\delta u_{i,j}^{\left( m \right)}{\text{d}}V}} \\
     {\text{ }}&{ = \iiint\limits_{{V^{\left( m \right)}}} {\left[ {{{\left( {\frac{{\partial {A^{\left( m \right)}}}}{{\partial \varepsilon _{ij}^{\left( m \right)}}}\delta u_i^{\left( m \right)}} \right)}_{,j}} - {{\left( {\frac{{\partial {A^{\left( m \right)}}}}{{\partial \varepsilon _{ij}^{\left( m \right)}}}} \right)}_{,j}}\delta u_i^{\left( m \right)}} \right]{\text{d}}V}} \\
     {\text{ }}&{ = \iint\limits_{{S^{\left( m \right)}}} {\frac{{\partial {A^{\left( m \right)}}}}{{\partial \varepsilon _{ij}^{\left( m \right)}}}n_j^{\left( m \right)}\delta u_i^{\left( m \right)}{\text{d}}S} - \iiint\limits_{{V^{\left( m \right)}}} {{{\left( {\frac{{\partial {A^{\left( m \right)}}}}{{\partial \varepsilon _{ij}^{\left( m \right)}}}} \right)}_{,j}}\delta u_i^{\left( m \right)}{\text{d}}V}} \\
     {\text{ }}&{ = \iint\limits_{S_p^{\left( m \right)}} {\frac{{\partial {A^{\left( m \right)}}}}{{\partial \varepsilon _{ij}^{\left( m \right)}}}n_j^{\left( m \right)}\delta u_i^{\left( m \right)}{\text{d}}S} + \iint\limits_{{S^{\left( {m{m^\prime }} \right)}}} {\frac{{\partial {A^{\left( m \right)}}}}{{\partial \varepsilon _{ij}^{\left( m \right)}}}n_j^{\left( m \right)}\delta u_i^{\left( m \right)}{\text{d}}S} - \iiint\limits_{{V^{\left( m \right)}}} {{{\left( {\frac{{\partial {A^{\left( m \right)}}}}{{\partial \varepsilon _{ij}^{\left( m \right)}}}} \right)}_{,j}}\delta u_i^{\left( m \right)}{\text{d}}V}}
   \end{array}
   \end{equation}

其中，\ :math:`{{S^{\left( m \right)}}}`\ 由三部分组成

.. math::


   \begin{equation}
   {S^{(m)}} = S_p^{(m)} + S_u^{(m)} + {S^{\left( {m{m^\prime }} \right)}}
   \end{equation}

:math:`{S^{\left( {m{m^\prime }} \right)}}`\ 是相邻有限单元 :math:`m` 与
:math:`m^{\prime}`
之间的交界面。根据位移协调的条件，在单元交界面\ :math:`{S^{\left( {m{m^\prime }} \right)}}`
上有

.. math::


   \begin{equation}
   u_i^{\left( m \right)} = u_i^{\left( {{m^\prime }} \right)} = u_i^{\left( {m{m^\prime }} \right)} \quad\left(\text { 在 } S^{\left(m m^{\prime}\right)} \text { 上 }\right)
   \end{equation}

或

.. math::


   \begin{equation}
   \delta u_i^{\left( m \right)} = \delta u_i^{\left( {{m^\prime }} \right)} = \delta u_i^{\left( {m{m^\prime }} \right)} \quad\left(\text { 在 } S^{\left(m m^{\prime}\right)} \text { 上 }\right)
   \end{equation}

因此，式\ :math:numref:`eq:delta_Pi*_1`\ 可写成

.. math::


   \begin{equation}
   \begin{array}{*{20}{l}}
     {\delta {\Pi ^*}}&{ = \sum\limits_{m = 1}^N {\left\{ { - \iiint\limits_{{V^{\left( m \right)}}} {\left[ {{{\left( {\frac{{\partial {A^{\left( m \right)}}}}{{\partial \varepsilon _{ij}^{\left( m \right)}}}} \right)}_{,j}} + {f_i}} \right]\delta u_i^{\left( m \right)}{\text{d}}V} + \iint\limits_{S_p^{\left( m \right)}} {\left( {\frac{{\partial {A^{\left( m \right)}}}}{{\partial \varepsilon _{ij}^{\left( m \right)}}}n_j^{\left( m \right)} - {{\bar p}_i}} \right)\delta u_i^{\left( m \right)}{\text{d}}S}} \right\}} } \\
     {\text{ }}&{ + \sum\limits_{\left( {m{m^\prime }} \right)} {\iint\limits_{{S^{\left( {m{m^\prime }} \right)}}} {\left( {\frac{{\partial {A^{\left( m \right)}}}}{{\partial \varepsilon _{ij}^{\left( m \right)}}}n_j^{\left( m \right)} + \frac{{\partial {A^{\left( {{m^\prime }} \right)}}}}{{\partial \varepsilon _{ij}^{\left( {{m^\prime }} \right)}}}n_j^{\left( {{m^\prime }} \right)}} \right)\delta u_i^{\left( {m{m^\prime }} \right)}{\text{d}}S}} }
   \end{array}
   \end{equation}

由于 :math:`\delta u_{i}^{(m)}` 在 :math:`{{V^{\left( m \right)}}}`
中和在 :math:`S_p^{\left( m \right)}`
上，\ :math:`\delta u_{i}^{\left(m m^{\prime}\right)}`
在相邻有限元之间的交界面上，都是独立的变量，所以
:math:`\delta \Pi^{*}=0` 给出了下列关系

.. math::


   \begin{equation}
   {\left( {\frac{{\partial {A^{\left( m \right)}}}}{{\partial \varepsilon _{ij}^{\left( m \right)}}}} \right)_{,j}} + {f_i} = 0 \quad \left( \text { 在 } V^{\left( m \right)} \text { 内 } \right)
   \end{equation}

.. math::


   \begin{equation}
   \frac{{\partial {A^{\left( m \right)}}}}{{\partial \varepsilon _{ij}^{\left( m \right)}}}n_j^{\left( m \right)} - {{\bar p}_i} = 0 \quad \left( \text { 在 } S_p^{\left( m \right)} \text { 上 } \right)
   \end{equation}

.. math::


   \begin{equation}
   \frac{{\partial {A^{\left( m \right)}}}}{{\partial \varepsilon _{ij}^{\left( m \right)}}}n_j^{\left( m \right)} + \frac{{\partial {A^{\left( {{m^\prime }} \right)}}}}{{\partial \varepsilon _{ij}^{\left( {{m^\prime }} \right)}}}n_j^{\left( {{m^\prime }} \right)} = 0 \quad \left( \text { 在 } S^{\left(m m^{\prime}\right)} \text { 上 } \right)
   \end{equation}

应用物理方程\ :math:`\frac{{\partial {A^{\left( m \right)}}}}{{\partial \varepsilon _{ij}^{\left( m \right)}}} = \sigma _{ij}^{\left( m \right)}`\ ，可得

.. math::
   \begin{equation}
   \sigma _{ij,j}^{\left( m \right)} + {f_i} = 0 \quad \left( \text { 在 } V^{\left( m \right)} \text { 内 } \right)
   \label{eq:equilibrium_element_1}
   \end{equation}
   :label: eq:equilibrium_element_1

.. math::
   \begin{equation}
   \sigma _{ij}^{\left( m \right)} n_j - {{\bar p}_i} = 0 \quad \left( \text { 在 } S_p^{\left( m \right)} \text { 上 } \right)
   \label{eq:sp_element_1}
   \end{equation}
   :label: eq:sp_element_1

.. math::
   \begin{equation}
   \sigma _{ij}^{\left( m \right)}n_j^{\left( m \right)} + \sigma _{ij}^{\left( m^{\prime} \right)}n_j^{\left( {{m^\prime }} \right)} = 0 \quad \left( \text { 在 } S^{\left(m m^{\prime}\right)} \text { 上 } \right)
   \label{eq:stress_continuity_1}
   \end{equation}
   :label: eq:stress_continuity_1

这就是位移协调元的最小势能原理，以上各式表明，\ :math:`\Pi^{*}`
取极值等效于弹性体各单元的平衡方程（式\ :math:numref:`eq:equilibrium_element_1`\ ）和单元边界上的力边界条件（式\ :math:numref:`eq:sp_element_1`\ ），而且给出了相邻单元交界面上应力矢量的连续条件（式\ :math:numref:`eq:stress_continuity_1`\ ）。值得指出的是，“在相邻单元的交界面上应力矢量是连续的”这一结论，它的前提是假定所选择的单元位移函数，不仅在单元交界面上是协调的，而且要使它满足有限元平衡方程（式\ :math:numref:`eq:equilibrium_element_1`\ ）和外力已知边界条件（式\ :math:numref:`eq:sp_element_1`\ ），也就是有限元平衡方程和外力已知边界条件不致遭到破坏。
