基于位移协调元最小势能原理的有限元方程
================================================================================

通过最小势能原理，我们把微分方程边值问题化为泛函极值问题，本节将通过该泛函极值问题近似解法，推导出基于位移协调元的最小势能原理的有限元方程。

根据变分法，系统的总势能\ :math:`\Pi`\ ，是关于状态变量\ :math:`u_i`\ 的泛函；\ :math:`u_i`\ 是未知的场函数，它的解一定使得总势能\ :math:`\Pi`\ 最小，即\ :math:`\Pi`\ 的变分\ :math:`\delta \Pi=0`\ （最小势能原理）。这似乎给我们提供了一个求解的思路：只要我们从众多的函数中，找到一个函数\ :math:`u_i`\ 使\ :math:`\delta \Pi=0`\ ，那么\ :math:`u_i`\ 就是问题的解。但不幸的是：我们不可能把所有可能的函数罗列出来，从中找到满足\ :math:`\delta \Pi=0`\ 的解。因为，可能的函数有无限多个而且毫无规律，从中找到答案可能比“大海捞针”更加困难。

里兹法的提出，让变分法的求解思路得以实现。里兹法的核心思想是：放弃寻找准确解，而从一簇特定的函数中找到最接近准确解的近似解。这一簇“特定的函数”通常称为“近似函数”或者“试函数”（trial
functions）。采用里兹法求近似解时，我们可以自由地选择一类便于处理的函数（通常是多项式函数）作为试函数，然后从这些试函数中找到最佳的近似解。同时考虑到对于位移协调元，在单元交界面\ :math:`{S^{\left( {m{m^\prime }} \right)}}`\ 上有

.. math::
   \begin{equation}
   u_i^{\left( m \right)} = u_i^{\left( {{m^\prime }} \right)} = u_i^{\left( {m{m^\prime }} \right)} \quad\left(\text { 在 } S^{\left(m m^{\prime}\right)} \text { 上 }\right)
   \label{eq:displacement_coordination_1}
   \end{equation}
   :label: eq:displacement_coordination_1

因此我们通过节点插值的方法来选取单元试函数，假设其形式为

.. math::
   \begin{equation}
   u_i^{\left( m \right)} \approx \hat u_i^{\left( m \right)} = N_k^{\left( m \right)}q_{ik}^{\left( m \right)}
   \end{equation}

其中，\ :math:`N_k^{\left( m \right)}`\ 是单元试函数的基函数，在有限元中通常称为形函数，需要注意的是\ :math:`N_k^{\left( m \right)}`\ 只是坐标\ :math:`x_i`\ 的函数，\ :math:`q_{ik}^{\left( m \right)}`\ 是待定系数，也是单元的节点位移，其中下标\ :math:`k`\ 代表单元的节点数目，\ :math:`i`\ 为坐标维度。选择合适的基函数\ :math:`N_k^{\left( m \right)}`\ ，使试函数满足直接（位移）边界条件（无需满足自然（力）边界条件，因为自然边界条件隐含在泛函总势能中）。此时如果相邻单元选取相同的节点位移插值基函数\ :math:`N_k^{\left( m \right)}`\ ，则在单元交界面\ :math:`{S^{\left( {m{m^\prime }} \right)}}`\ 上自然满足式\ :math:numref:`eq:displacement_coordination_1`\ 所要求的位移协调条件。对于线弹性小变形为题，我们将所有单元试函数\ :math:`\hat u_i^{\left( m \right)}`\ 带入式\ :math:numref:`eq:Pi*_1`\ 可得

.. math::
   \begin{equation}
   {{\hat \Pi }^*} = \sum\limits_{m = 1}^N {\left\{ {\iiint\limits_{{V^{\left( m \right)}}} {\left[ {\frac{1}{2}{E_{ijkl}}\varepsilon _{ij}^{\left( m \right)}\varepsilon _{kl}^{\left( m \right)} - {f_i}N_k^{\left( m \right)}q_{ik}^{\left( m \right)}} \right]{\text{d}}V} - \iint\limits_{S_p^{\left( m \right)}} {{{\bar p}_i}N_k^{\left( m \right)}q_{ik}^{\left( m \right)}{\text{d}}S}} \right\}}
   \label{eq:hat_Pi*_1}
   \end{equation}
   :label: eq:hat_Pi*_1

现在，原问题变为：找到一组合适的节点位移\ :math:`q_{in}`\ （\ :math:`n`\ 为系统离散后的节点总数），代入所有单元试函数\ :math:`\hat u_i^{\left( m \right)}`\ ，使得总势能\ :math:`{\hat \Pi }^*`\ 取最小值。需要注意的是此时泛函\ :math:`{\Pi }^*`\ 的极值问题转变为了函数\ :math:`{{\hat \Pi }^*}`\ 的极值问题，此时\ :math:`{{\hat \Pi }^*}`\ 只是节点坐标\ :math:`q_{in}`\ 的函数，对于三维问题共有\ :math:`3\times i \times n`\ 个未知数。用数学语言来描述，即：

.. math::
   \begin{equation}
   \frac{{\partial {{\hat \Pi }^*}\left( {{q_{in}}} \right)}}{{\partial {q_{in}}}} = 0
   \end{equation}

可以得到\ :math:`3\times i \times n`\ 个代数方程。联立方程组，可以求出这些节点位移\ :math:`q_{in}`\ ；代入单元试函数\ :math:`\hat u_i^{\left( m \right)}`\ ，即可得到近似解。

在有限元方法中，根据应力和应变张量的对称性，采用Voigt标记将张量符号表示为矩阵乘法，其中\ :math:`\sigma_{ij}`\ 和\ :math:`\varepsilon_{ij}`\ 分别表示为列向量：

.. math::
   \begin{equation}
   {\mathbf{\varepsilon }} = \left\{ {\begin{array}{*{20}{c}}
     {{\varepsilon _{11}}} \\
     {{\varepsilon _{22}}} \\
     {{\varepsilon _{33}}} \\
     {{\varepsilon _{12}}} \\
     {{\varepsilon _{23}}} \\
     {{\varepsilon _{31}}}
   \end{array}} \right\}, \quad {\mathbf{\sigma }} = \left\{ {\begin{array}{*{20}{c}}
     {{\sigma _{11}}} \\
     {{\sigma _{22}}} \\
     {{\sigma _{33}}} \\
     {{\sigma _{12}}} \\
     {{\sigma _{23}}} \\
     {{\sigma _{31}}}
   \end{array}} \right\}
   \end{equation}

线弹性的物理方程\ :math:`{\sigma _{ij}} = {E_{ijkl}}{\varepsilon _{ij}}`\ 的矩阵形式为：

.. math::
   \begin{equation}
   {\mathbf{\sigma }} = {\mathbf{D\varepsilon }}
   \end{equation}

其中\ :math:`{\mathbf{D}}`\ 为弹性矩阵。
单元位移函数\ :math:`u_i`\ 表示为列向量：

.. math::
   \begin{equation}
   {\mathbf{u}} = \left\{ {\begin{array}{*{20}{c}}
     {{u_1}} \\
     {{u_2}} \\
     {{u_3}}
   \end{array}} \right\}
   \end{equation}

定义微分算子矩阵

.. math::
   \begin{equation}
   {\mathbf{L}} = \left[ {\begin{array}{*{20}{c}}
     {\frac{\partial }{{\partial {x_1}}}}&0&0 \\
     0&{\frac{\partial }{{\partial {x_2}}}}&0 \\
     0&0&{\frac{\partial }{{\partial {x_3}}}} \\
     {\frac{1}{2}\frac{\partial }{{\partial {x_2}}}}&{\frac{1}{2}\frac{\partial }{{\partial {x_1}}}}&0 \\
     0&{\frac{1}{2}\frac{\partial }{{\partial {x_3}}}}&{\frac{1}{2}\frac{\partial }{{\partial {x_2}}}} \\
     {\frac{1}{2}\frac{\partial }{{\partial {x_3}}}}&0&{\frac{1}{2}\frac{\partial }{{\partial {x_1}}}}
   \end{array}} \right]
   \end{equation}

则几何方程\ :math:`{\varepsilon _{ij}} = \frac{1}{2}\left( {{u_{i,j}} + {u_{j,i}}} \right)`\ 可以表示为矩阵乘法形式：

.. math::
   \begin{equation}
   \left\{ {\begin{array}{*{20}{c}}
     {{\varepsilon _{11}}} \\
     {{\varepsilon _{22}}} \\
     {{\varepsilon _{33}}} \\
     {{\varepsilon _{12}}} \\
     {{\varepsilon _{23}}} \\
     {{\varepsilon _{31}}}
   \end{array}} \right\} = \left[ {\begin{array}{*{20}{c}}
     {\frac{\partial }{{\partial {x_1}}}}&0&0 \\
     0&{\frac{\partial }{{\partial {x_2}}}}&0 \\
     0&0&{\frac{\partial }{{\partial {x_3}}}} \\
     {\frac{1}{2}\frac{\partial }{{\partial {x_2}}}}&{\frac{1}{2}\frac{\partial }{{\partial {x_1}}}}&0 \\
     0&{\frac{1}{2}\frac{\partial }{{\partial {x_3}}}}&{\frac{1}{2}\frac{\partial }{{\partial {x_2}}}} \\
     {\frac{1}{2}\frac{\partial }{{\partial {x_3}}}}&0&{\frac{1}{2}\frac{\partial }{{\partial {x_1}}}}
   \end{array}} \right]\left\{ {\begin{array}{*{20}{c}}
     {{u_1}} \\
     {{u_2}} \\
     {{u_3}}
   \end{array}} \right\}
   \end{equation}

记为

.. math::
   \begin{equation}
   {\mathbf{\varepsilon }} = {\mathbf{Lu}}
   \label{eq:epsilon_Lu}
   \end{equation}
   :label: eq:epsilon_Lu

将节点位移矢量\ :math:`q_{ik}`\ 表示为列向量：

.. math::
   \begin{equation}
   {\mathbf{q}} = \left\{ {\begin{array}{*{20}{c}}
     {{q_{11}}} \\
     {{q_{21}}} \\
     {{q_{31}}} \\
     {{q_{12}}} \\
     {{q_{22}}} \\
     {{q_{32}}} \\
      \vdots  \\
     {{q_{1k}}} \\
     {{q_{2k}}} \\
     {{q_{3k}}}
   \end{array}} \right\}
   \end{equation}

插值基函数\ :math:`N_k`\ 表示为矩阵形式：

.. math::
   \begin{equation}
   {\mathbf{N}} = \left[ {\begin{array}{*{20}{c}}
     {{N_1}}&0&0&{{N_2}}&0&0& \cdots &{{N_k}}&0&0 \\
     0&{{N_1}}&0&0&{{N_2}}&0& \cdots &0&{{N_k}}&0 \\
     0&0&{{N_1}}&0&0&{{N_2}}& \cdots &0&0&{{N_k}}
   \end{array}} \right] = \left[ {\begin{array}{*{20}{c}}
     {{\mathbf{I}}{N_1}}&{{\mathbf{I}}{N_2}}&{{\mathbf{I}}{N_3}}& \cdots &{{\mathbf{I}}{N_k}}
   \end{array}} \right]
   \end{equation}

则可以得到

.. math::
   \begin{equation}
   \left\{ {\begin{array}{*{20}{c}}
     {{u_1}} \\
     {{u_2}} \\
     {{u_3}}
   \end{array}} \right\} = \left[ {\begin{array}{*{20}{c}}
     {{N_1}}&0&0&{{N_2}}&0&0& \cdots &{{N_k}}&0&0 \\
     0&{{N_1}}&0&0&{{N_2}}&0& \cdots &0&{{N_k}}&0 \\
     0&0&{{N_1}}&0&0&{{N_2}}& \cdots &0&0&{{N_k}}
   \end{array}} \right]\left\{ {\begin{array}{*{20}{c}}
     {{q_{11}}} \\
     {{q_{21}}} \\
     {{q_{31}}} \\
     {{q_{12}}} \\
     {{q_{22}}} \\
     {{q_{32}}} \\
      \vdots  \\
     {{q_{1k}}} \\
     {{q_{2k}}} \\
     {{q_{3k}}}
   \end{array}} \right\}
   \end{equation}

记为

.. math::
   \begin{equation}
   {\mathbf{u}} = {\mathbf{Nq}}
   \label{eq:u_Nq}
   \end{equation}
   :label: eq:u_Nq

因此我们可以得到以下关系

.. math::
   \begin{equation}
   {\mathbf{\varepsilon }} = {\mathbf{Lu}} = {\mathbf{LNq}} = {\mathbf{Bq}}
   \label{eq:epsilon_Bq}
   \end{equation}
   :label: eq:epsilon_Bq

其中

.. math::
   {\mathbf{B}} = {\mathbf{LN}} = \left[ {\begin{array}{*{20}{c}}
     {\frac{\partial }{{\partial {x_1}}}}&0&0 \\
     0&{\frac{\partial }{{\partial {x_2}}}}&0 \\
     0&0&{\frac{\partial }{{\partial {x_3}}}} \\
     {\frac{1}{2}\frac{\partial }{{\partial {x_2}}}}&{\frac{1}{2}\frac{\partial }{{\partial {x_1}}}}&0 \\
     0&{\frac{1}{2}\frac{\partial }{{\partial {x_3}}}}&{\frac{1}{2}\frac{\partial }{{\partial {x_2}}}} \\
     {\frac{1}{2}\frac{\partial }{{\partial {x_3}}}}&0&{\frac{1}{2}\frac{\partial }{{\partial {x_1}}}}
   \end{array}} \right]\left[ {\begin{array}{*{20}{c}}
     {{N_1}}&0&0&{{N_2}}&0&0& \cdots &{{N_k}}&0&0 \\
     0&{{N_1}}&0&0&{{N_2}}&0& \cdots &0&{{N_k}}&0 \\
     0&0&{{N_1}}&0&0&{{N_2}}& \cdots &0&0&{{N_k}}
   \end{array}} \right]

.. math::
   \begin{equation}
    = \left[ {\begin{array}{*{20}{c}}
     {\frac{{\partial {N_1}}}{{\partial {x_1}}}}&0&0&{\frac{{\partial {N_2}}}{{\partial {x_1}}}}&0&0& \cdots &{\frac{{\partial {N_k}}}{{\partial {x_1}}}}&0&0 \\
     0&{\frac{{\partial {N_1}}}{{\partial {x_2}}}}&0&0&{\frac{{\partial {N_2}}}{{\partial {x_2}}}}&0& \cdots &0&{\frac{{\partial {N_k}}}{{\partial {x_2}}}}&0 \\
     0&0&{\frac{{\partial {N_1}}}{{\partial {x_3}}}}&0&0&{\frac{{\partial {N_2}}}{{\partial {x_3}}}}& \cdots &0&0&{\frac{{\partial {N_k}}}{{\partial {x_3}}}} \\
     {\frac{1}{2}\frac{{\partial {N_1}}}{{\partial {x_2}}}}&{\frac{1}{2}\frac{{\partial {N_1}}}{{\partial {x_1}}}}&0&{\frac{1}{2}\frac{{\partial {N_2}}}{{\partial {x_2}}}}&{\frac{1}{2}\frac{{\partial {N_2}}}{{\partial {x_1}}}}&0& \cdots &{\frac{1}{2}\frac{{\partial {N_k}}}{{\partial {x_2}}}}&{\frac{1}{2}\frac{{\partial {N_k}}}{{\partial {x_1}}}}&0 \\
     0&{\frac{1}{2}\frac{{\partial {N_1}}}{{\partial {x_3}}}}&{\frac{1}{2}\frac{{\partial {N_1}}}{{\partial {x_2}}}}&0&{\frac{1}{2}\frac{{\partial {N_2}}}{{\partial {x_3}}}}&{\frac{1}{2}\frac{{\partial {N_2}}}{{\partial {x_2}}}}& \cdots &0&{\frac{1}{2}\frac{{\partial {N_k}}}{{\partial {x_3}}}}&{\frac{1}{2}\frac{{\partial {N_k}}}{{\partial {x_2}}}} \\
     {\frac{1}{2}\frac{{\partial {N_1}}}{{\partial {x_3}}}}&0&{\frac{1}{2}\frac{{\partial {N_1}}}{{\partial {x_1}}}}&{\frac{1}{2}\frac{{\partial {N_2}}}{{\partial {x_3}}}}&0&{\frac{1}{2}\frac{{\partial {N_2}}}{{\partial {x_1}}}}& \cdots &{\frac{1}{2}\frac{{\partial {N_k}}}{{\partial {x_3}}}}&0&{\frac{1}{2}\frac{{\partial {N_k}}}{{\partial {x_1}}}}
   \end{array}} \right]
   \end{equation}

单元的体力函数\ :math:`f_i`\ 和力边界条件\ :math:`{{{\bar p}_i}}`\ 分别表示成列向量

.. math::
   \begin{equation}
   {\mathbf{f}} = \left\{ {\begin{array}{*{20}{c}}
     {{f_1}} \\
     {{f_2}} \\
     {{f_3}}
   \end{array}} \right\}, \quad {\mathbf{\bar p}} = \left\{ {\begin{array}{*{20}{c}}
     {{{\bar p}_1}} \\
     {{{\bar p}_2}} \\
     {{{\bar p}_3}}
   \end{array}} \right\}
   \end{equation}

则式\ :math:numref:`eq:hat_Pi*_1`\ 中第\ :math:`m`\ 号单元对应的势能函数可以表示为

.. math::
   \begin{equation}
   {{\hat \Pi }^{*\left( {\text{m}} \right)}} = \iiint\limits_{{V^{\left( m \right)}}} {\left[ {\frac{1}{2}{{\left( {{{\mathbf{\varepsilon }}^{\left( m \right)}}} \right)}^{\text{T}}}{\mathbf{D}}{{\mathbf{\varepsilon }}^{\left( m \right)}} - {{\left( {{{\mathbf{q}}^{\left( m \right)}}} \right)}^{\text{T}}}{{\left( {{{\mathbf{N}}^{\left( m \right)}}} \right)}^{\text{T}}}{\mathbf{f}}} \right]{\text{d}}V} - \iint\limits_{S_p^{\left( m \right)}} {{{\left( {{{\mathbf{q}}^{\left( m \right)}}} \right)}^{\text{T}}}{{\left( {{{\mathbf{N}}^{\left( m \right)}}} \right)}^{\text{T}}}{{{\mathbf{\bar p}}}^{\text{T}}}{\text{d}}S}
   \end{equation}

带入式\ :math:numref:`eq:epsilon_Lu`\ ，\ :math:numref:`eq:u_Nq`\ ，和\ :math:numref:`eq:epsilon_Bq`\ 得

.. math::
   \begin{equation}
   {{\hat \Pi }^{*\left( {\text{m}} \right)}} = \iiint\limits_{{V^{\left( m \right)}}} {\left[ {\frac{1}{2}{{\left( {{{\mathbf{q}}^{\left( m \right)}}} \right)}^{\text{T}}}{{\left( {{{\mathbf{B}}^{\left( m \right)}}} \right)}^{\text{T}}}{\mathbf{D}}{{\mathbf{B}}^{\left( m \right)}}{{\mathbf{q}}^{\left( m \right)}} - {{\left( {{{\mathbf{q}}^{\left( m \right)}}} \right)}^{\text{T}}}{{\left( {{{\mathbf{N}}^{\left( m \right)}}} \right)}^{\text{T}}}{\mathbf{f}}} \right]{\text{d}}V} - \iint\limits_{S_p^{\left( m \right)}} {{{\left( {{{\mathbf{q}}^{\left( m \right)}}} \right)}^{\text{T}}}{{\left( {{{\mathbf{N}}^{\left( m \right)}}} \right)}^{\text{T}}}{{{\mathbf{\bar p}}}^{\text{T}}}{\text{d}}S}
   \end{equation}

因为\ :math:`{{{\mathbf{q}}^{\left( m \right)}}}`\ 是单元对应节点坐标，与积分运算无关，整理可得

.. math::
   \begin{equation}
   {{\hat \Pi }^{*\left( {\text{m}} \right)}} = \frac{1}{2}{\left( {{{\mathbf{q}}^{\left( m \right)}}} \right)^{\text{T}}}\left[ {\iiint\limits_{{V^{\left( m \right)}}} {{{\left( {{{\mathbf{B}}^{\left( m \right)}}} \right)}^{\text{T}}}{\mathbf{D}}{{\mathbf{B}}^{\left( m \right)}}{\text{d}}V}} \right]{{\mathbf{q}}^{\left( m \right)}} - {\left( {{{\mathbf{q}}^{\left( m \right)}}} \right)^{\text{T}}}\left[ {\iiint\limits_{{V^{\left( m \right)}}} {{{\left( {{{\mathbf{N}}^{\left( m \right)}}} \right)}^{\text{T}}}{\mathbf{f}}{\text{d}}V + \iint\limits_{S_p^{\left( m \right)}} {{{\left( {{{\mathbf{N}}^{\left( m \right)}}} \right)}^{\text{T}}}{{{\mathbf{\bar p}}}^{\text{T}}}{\text{d}}S}}} \right]
   \end{equation}

记为

.. math::
   \begin{equation}
   {{\hat \Pi }^{*\left( {\text{m}} \right)}} = \frac{1}{2}{\left( {{{\mathbf{q}}^{\left( m \right)}}} \right)^{\text{T}}}{{\mathbf{K}}^{\left( m \right)}}{{\mathbf{q}}^{\left( m \right)}} - {\left( {{{\mathbf{q}}^{\left( m \right)}}} \right)^{\text{T}}}{{\mathbf{R}}^{\left( m \right)}}
   \end{equation}

其中

.. math::
   \begin{equation}
   {{\mathbf{K}}^{\left( m \right)}} = \iiint\limits_{{V^{\left( m \right)}}} {{{\mathbf{B}}^{\left( m \right)}}^{\text{T}}{\mathbf{D}}{{\mathbf{B}}^{\left( m \right)}}{\text{d}}V}
   \end{equation}

.. math::
   \begin{equation}
   {{\mathbf{R}}^{\left( m \right)}} = \iiint\limits_{{V^{\left( m \right)}}} {{{\left( {{{\mathbf{N}}^{\left( m \right)}}} \right)}^{\text{T}}}{\mathbf{f}}{\text{d}}V + \iint\limits_{S_p^{\left( m \right)}} {{{\left( {{{\mathbf{N}}^{\left( m \right)}}} \right)}^{\text{T}}}{{{\mathbf{\bar p}}}^{\text{T}}}{\text{d}}S}}
   \end{equation}

不难得知\ :math:`{{\mathbf{K}}^{\left( m \right)}}`\ 就单元的刚度矩阵，\ :math:`\frac{1}{2}{\left( {{{\mathbf{q}}^{\left( m \right)}}} \right)^{\text{T}}}{{\mathbf{K}}^{\left( m \right)}}{{\mathbf{q}}^{\left( m \right)}}`\ 为单元的应变能，\ :math:`{{\mathbf{R}}^{\left( m \right)}}`\ 为单元的节点外力矢量。
式\ :math:numref:`eq:hat_Pi*_1`\ 代表的系统总势能可以表示为

.. math::
   \begin{equation}
   {{\hat \Pi }^*} = \sum\limits_{m = 1}^N {\left\{ {\frac{1}{2}{{\left( {{{\mathbf{q}}^{\left( m \right)}}} \right)}^{\text{T}}}{{\mathbf{K}}^{\left( m \right)}}{{\mathbf{q}}^{\left( m \right)}} - {{\left( {{{\mathbf{q}}^{\left( m \right)}}} \right)}^{\text{T}}}{{\mathbf{R}}^{\left( m \right)}}} \right\}}
   \end{equation}

采用有限元中的组集方法，将不同单元的相同节点位移进行合并，可以得到总体的矩阵方程

.. math::
   \begin{equation}
   {{\hat \Pi }^*} = \frac{1}{2}{{\mathbf{q}}^{\text{T}}}{\mathbf{Kq}} - {{\mathbf{q}}^{\text{T}}}{\mathbf{R}}
   \end{equation}

其中，\ :math:`\mathbf{q}`\ ，\ :math:`\mathbf{K}`\ ，\ :math:`\mathbf{R}`\ 分别为整体的节点位移矢量，刚度矩阵和外力矢量。根据函数的极值条件\ :math:`\frac{{\partial {{\hat \Pi }^*}}}{{\partial {\mathbf{q}}}} = 0`\ 可得

.. math::
   \begin{equation}
   {\mathbf{Kq}} - {\mathbf{R}} = 0
   \end{equation}
