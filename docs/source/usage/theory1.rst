最小势能原理
================================================================================

:numref:`fig:logo`

一个小变形线弹性体静力学问题，共有
:math:`\sigma_{ij}`\ ，\ :math:`\varepsilon_{ij}` 和 :math:`u_{i}`
等15个待定函数，它们在域\ :math:`V`\ 中必须满足弹性力学的15个基本方程和6个边界条件。现在，我们尝试将上述问题等效为泛函的驻值问题，建立最小势能原理。

最小势能原理：在满足几何方程\ :math:`{\varepsilon _{ij}} = \frac{1}{2}\left( {{u_{i,j}} + {u_{j,i}}} \right)`\ 和位移边界条件\ :math:`u_{i}=\overline {u}_{i}`\ 的所有允许位移函数中，实际的位移\ :math:`u_i`\ 必定可以使弹性体的总势能

.. math::
   \begin{equation}
   \Pi  = \iiint\limits_V {A\left( {{\varepsilon _{ij}}} \right){\text{d}}V} - \iiint\limits_V {{f_i}{u_i}{\text{d}}V - \iint\limits_{{S_p}} {{{\bar p}_i}{u_i}{\text{d}}S}}
   \label{eq:Pi_1}
   \end{equation}
   :label: eq:Pi_1

为最小。

为了证明最小位能原理，先求\ :math:`\Pi`\ 的一阶变分：

.. math::
   \begin{equation}
   \delta \Pi  = \iiint\limits_V {\frac{{\partial A}}{{\partial {\varepsilon _{ij}}}}\delta {\varepsilon _{ij}}{\text{d}}V} - \iiint\limits_V {{f_i}\delta {u_i}{\text{d}}V - \iint\limits_{{S_p}} {{{\bar p}_i}\delta {u_i}{\text{d}}S}}
   \label{eq:delta_Pi}
   \end{equation}
   :label: eq:delta_Pi
   :name: eq:delta_Pi

由于几何方程事先要满足，同时基于应力的对称性\ :math:`\sigma_{ij}=\sigma_{ji}`\ ，有

.. math::
   \begin{equation}
   \frac{{\partial A}}{{\partial {\varepsilon _{ij}}}}\delta {\varepsilon _{ij}} = \frac{{\partial A}}{{\partial {\varepsilon _{ij}}}}\frac{1}{2}\delta \left( {{u_{i,j}} + {u_{j,i}}} \right) = \frac{1}{2}{\sigma _{ij}}\delta {u_{i,j}} + \frac{1}{2}{\sigma _{ij}}\delta {u_{j,i}} = {\sigma _{ij}}\delta {u_{i,j}} = \frac{{\partial A}}{{\partial {\varepsilon _{ij}}}}\delta {u_{i,j}}
   \end{equation}

根据求导的链式法则有

.. math::
   \begin{equation}
   {\left( {\frac{{\partial A}}{{\partial {\varepsilon _{ij}}}}\delta {u_i}} \right)_{,j}} = {\left( {\frac{{\partial A}}{{\partial {\varepsilon _{ij}}}}} \right)_{,j}}\delta {u_i} + \frac{{\partial A}}{{\partial {\varepsilon _{ij}}}}\delta {u_{i,j}}
   \end{equation}

同时根据高斯散度定理可知

.. math::
   \begin{equation}
   \iiint\limits_V {{{\left( {\frac{{\partial A}}{{\partial {\varepsilon _{ij}}}}\delta {u_i}} \right)}_{,j}}{\text{d}}V} = \iint\limits_S {\frac{{\partial A}}{{\partial {\varepsilon _{ij}}}}\delta {u_i}{n_j}{\text{d}}S}
   \end{equation}

因此式\ :math:numref:`eq:delta_Pi`\  （ :hoverxref:`查看公式 <eq:delta_Pi>` ）的右边第一项

.. math::
   \begin{equation}
   \begin{array}{*{20}{l}}
     {\iiint\limits_V {\frac{{\partial A}}{{\partial {\varepsilon _{ij}}}}\delta {\varepsilon _{ij}}{\text{d}}V}}&{ = \iiint\limits_V {\frac{{\partial A}}{{\partial {\varepsilon _{ij}}}}\delta {u_{i,j}}{\text{d}}V}} \\
     {\text{ }}&{ = \iiint\limits_V {\left[ {{{\left( {\frac{{\partial A}}{{\partial {\varepsilon _{ij}}}}\delta {u_i}} \right)}_{,j}} - {{\left( {\frac{{\partial A}}{{\partial {\varepsilon _{ij}}}}} \right)}_{,j}}\delta {u_i}} \right]{\text{d}}V}} \\
     {\text{ }}&{ = \iint\limits_S {\frac{{\partial A}}{{\partial {\varepsilon _{ij}}}}\delta {u_i}{n_j}{\text{d}}S} - \iiint\limits_V {{{\left( {\frac{{\partial A}}{{\partial {\varepsilon _{ij}}}}} \right)}_{,j}}\delta {u_i}{\text{d}}V}} \\
     {\text{ }}&{ = \iint\limits_{{S_p}} {\frac{{\partial A}}{{\partial {\varepsilon _{ij}}}}{n_j}\delta {u_i}{\text{d}}S} - \iiint\limits_V {{{\left( {\frac{{\partial A}}{{\partial {\varepsilon _{ij}}}}} \right)}_{,j}}\delta {u_i}{\text{d}}V}}
   \end{array}
   \end{equation}

上式中我们应用了条件，在\ :math:`S_u`\ 边界上\ :math:`\delta u_i=0`\ 。带入式\ :math:numref:`eq:delta_Pi`\ 得

.. math::
   \begin{equation}
   \delta \Pi  = \iint\limits_{{S_p}} {\frac{{\partial A}}{{\partial {\varepsilon _{ij}}}}{n_j}\delta {u_i}{\text{d}}S} - \iiint\limits_V {{{\left( {\frac{{\partial A}}{{\partial {\varepsilon _{ij}}}}} \right)}_{,j}}\delta {u_i}{\text{d}}V} - \iiint\limits_V {{f_i}\delta {u_i}{\text{d}}V - \iint\limits_{{S_p}} {{{\bar p}_i}\delta {u_i}{\text{d}}S}}
   \end{equation}

整理可得

.. math::
   \begin{equation}
   \delta \Pi  =  - \iiint\limits_V {\left[ {{{\left( {\frac{{\partial A}}{{\partial {\varepsilon _{ij}}}}} \right)}_{,j}} + {f_i}} \right]\delta {u_i}{\text{d}}V} + \iint\limits_{{S_p}} {\left( {\frac{{\partial A}}{{\partial {\varepsilon _{ij}}}}{n_j} - {{\bar p}_i}} \right)\delta {u_i}{\text{d}}S}
   \end{equation}

弹性体的总势能\ :math:`\Pi`\ 取极值的条件是\ :math:`\delta \Pi=0`\ ，因此必须有

.. math::
   \begin{equation}
   {\left( {\frac{{\partial A}}{{\partial {\varepsilon _{ij}}}}} \right)_{,j}} + {f_i} = 0 \quad \left( \text { 在 } V \text { 内 } \right)
   \end{equation}

.. math::
   \begin{equation}
   \frac{{\partial A}}{{\partial {\varepsilon _{ij}}}}{n_j} - {{\bar p}_i} = 0 \quad \left( \text { 在 } S_p \text { 上 } \right)
   \end{equation}

应用物理方程\ :math:`\frac{{\partial A}}{{\partial {\varepsilon _{ij}}}} = {\sigma _{ij}}`\ ，可得

.. math::
   \begin{equation}
   {\sigma _{ij,j}} + {f_i} = 0 \quad \left( \text { 在 } V \text { 内 } \right)
   \end{equation}

.. math::
   \begin{equation}
   {\sigma _{ij}}{n_j} - {{\bar p}_i} = 0 \quad \left( \text { 在 } S_p \text { 上 } \right)
   \end{equation}

以上两式就是平衡方程和力边界条件，即满足了\ :math:`\delta \Pi=0`\ ，就等于满足了以上两个条件。因此，使泛函\ :math:`\Pi`\ 取极值的的位移\ :math:`u_i`\ 就是真实的解。进一步我们可以证明该极值为最小值。
