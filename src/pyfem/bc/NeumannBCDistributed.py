# -*- coding: utf-8 -*-
"""

"""
from typing import Optional

from numpy import array, delete, dot, logical_and, ndarray, in1d, all, sqrt, zeros
from numpy.linalg import det

from pyfem.bc.BaseBC import BaseBC
from pyfem.io.Amplitude import Amplitude
from pyfem.io.BC import BC
from pyfem.io.Dof import Dof
from pyfem.io.Solver import Solver
from pyfem.isoelements.IsoElementShape import iso_element_shape_dict
from pyfem.isoelements.get_iso_element_type import get_iso_element_type
from pyfem.mesh.MeshData import MeshData
from pyfem.utils.colors import error_style


class NeumannBCDistributed(BaseBC):
    r"""
    Neumann边界条件：分布载荷。

    考虑在单元边界上施加某个分布载荷的情况，对于二维问题，面力 :math:`\mathbf{\bar p}` 在 :math:`\left( m \right)` 号单元所产生的等效节点载荷为：

    .. math::
        {{\mathbf{R}}^{\left( m \right)}} = \int\limits_{s_p^{\left( m \right)}} {{{\left( {{{\mathbf{N}}^{\left( m \right)}}} \right)}^{\text{T}}}{\mathbf{\bar p}}{\text{d}}s}

    对于三维问题，面力 :math:`\mathbf{\bar p}` 在 :math:`\left( m \right)` 号单元所产生的等效节点载荷为：

    .. math::
        {{\mathbf{R}}^{\left( m \right)}} = \iint\limits_{S_p^{\left( m \right)}} {{{\left( {{{\mathbf{N}}^{\left( m \right)}}} \right)}^{\text{T}}}{\mathbf{\bar p}}{\text{d}}S}

    因此，对于边界上的分布载荷，我们需要处理单元边界上的第一类曲线积分和第一类曲面积分。

    通过坐标变换，我们可以建立全局坐标和局部坐标的关系，二维下的弧微分可以表示为：

    .. math::
        {\text{d}}s = \sqrt {{{\left( {{\text{d}}x} \right)}^2} + {{\left( {{\text{d}}y} \right)}^2}}  = \sqrt {{{\left( {\frac{{\partial x}}{{\partial \xi }}{\text{d}}\xi  + \frac{{\partial x}}{{\partial \eta }}{\text{d}}\eta } \right)}^2} + {{\left( {\frac{{\partial y}}{{\partial \xi }}{\text{d}}\xi  + \frac{{\partial y}}{{\partial \eta }}{\text{d}}\eta } \right)}^2}}


    以四节点四边形等参元为例子::

      (-1,1)           (1,1)
        3---------------2
        |       η       |
        |       |       |
        |       o--ξ    |
        |               |
        |               |
        0---------------1
      (-1,-1)          (1,-1)

    边界上的第一类曲线积分可以表示为：

    .. math::
        {\left. {{{\mathbf{R}}^{\left( m \right)}}} \right|_{\xi  =  \pm 1}} = \int_{ - 1}^1 {{{\left( {{{\mathbf{N}}^{\left( m \right)}}} \right)}^{\text{T}}}{\mathbf{\bar p}}\sqrt {{{\left( {\frac{{\partial x}}{{\partial \eta }}} \right)}^2} + {{\left( {\frac{{\partial y}}{{\partial \eta }}} \right)}^2}} {\text{d}}\eta }

    .. math::
        {\left. {{{\mathbf{R}}^{\left( m \right)}}} \right|_{\eta  =  \pm 1}} = \int_{ - 1}^1 {{{\left( {{{\mathbf{N}}^{\left( m \right)}}} \right)}^{\text{T}}}{\mathbf{\bar p}}\sqrt {{{\left( {\frac{{\partial x}}{{\partial \xi }}} \right)}^2} + {{\left( {\frac{{\partial y}}{{\partial \xi }}} \right)}^2}} {\text{d}}\xi }

    通过选取合适的积分点，采用数值积分即可求得积分值。

    以三节点三角形等参元为例子::

       (0,1)
        2
        * *
        *   *
        *     *
        η       *
        |         *
        0---ξ * * * 1
       (0,0)       (1,0)

    边界上的第一类曲线积分可以表示为：

    .. math::
        {\left. {{{\mathbf{R}}^{\left( m \right)}}} \right|_{\xi  = 0}} = \int_0^1 {{{\left( {{{\mathbf{N}}^{\left( m \right)}}} \right)}^{\text{T}}}{\mathbf{\bar p}}\sqrt {{{\left( {\frac{{\partial x}}{{\partial \eta }}} \right)}^2} + {{\left( {\frac{{\partial y}}{{\partial \eta }}} \right)}^2}} {\text{d}}\eta }

        {\left. {{{\mathbf{R}}^{\left( m \right)}}} \right|_{\eta  = 0}} = \int_0^1 {{{\left( {{{\mathbf{N}}^{\left( m \right)}}} \right)}^{\text{T}}}{\mathbf{\bar p}}\sqrt {{{\left( {\frac{{\partial x}}{{\partial \xi }}} \right)}^2} + {{\left( {\frac{{\partial y}}{{\partial \xi }}} \right)}^2}} {\text{d}}\xi }

    注意，对于斜边 1-2，我们有直线方程 :math:`\eta  = 1 - \xi` ，此时 :math:`{\text{d}}\eta = -{\text{d}}\xi` ：

    .. math::
        {\left. {{{\mathbf{R}}^{\left( m \right)}}} \right|_{\eta  = 1 - \xi }} = \int_0^1 {{{\left( {{{\mathbf{N}}^{\left( m \right)}}} \right)}^{\text{T}}}{\mathbf{\bar p}}\sqrt {{{\left( {\frac{{\partial x}}{{\partial \xi }} - \frac{{\partial x}}{{\partial \eta }}} \right)}^2} + {{\left( {\frac{{\partial y}}{{\partial \xi }} - \frac{{\partial y}}{{\partial \eta }}} \right)}^2}} {\text{d}}\xi }

    但是这种表达式计算效率较低，同时难以和三维问题的表达式统一，因此我们对考虑将斜边 1-2 投影到其他坐标轴方向，由于斜边 1-2 的直线方程是已知的，因此有：

    .. math::
        {\left. {{{\mathbf{R}}^{\left( m \right)}}} \right|_{\eta  = 1 - \xi }} = \int_0^1 {{{\left( {{{\mathbf{N}}^{\left( m \right)}}} \right)}^{\text{T}}}{\mathbf{\bar p}}\sqrt {{{\left( {\frac{{\partial x}}{{\partial \xi }}} \right)}^2} + {{\left( {\frac{{\partial y}}{{\partial \xi }}} \right)}^2}} \sqrt {1 + {{\left( {\frac{{\partial \eta }}{{\partial \xi }}} \right)}^2}} {\text{d}}\xi }  = \int_0^1 {{{\left( {{{\mathbf{N}}^{\left( m \right)}}} \right)}^{\text{T}}}{\mathbf{\bar p}}\sqrt {{{\left( {\frac{{\partial x}}{{\partial \xi }}} \right)}^2} + {{\left( {\frac{{\partial y}}{{\partial \xi }}} \right)}^2}} \sqrt 2 {\text{d}}\xi }

    对于三维问题，我们有：

    .. math::
        {\text{d}}\vec \xi  = \frac{{\partial x}}{{\partial \xi }}{\text{d}}\xi \vec i + \frac{{\partial y}}{{\partial \xi }}{\text{d}}\xi \vec j + \frac{{\partial z}}{{\partial \xi }}{\text{d}}\xi \vec k

        {\text{d}}\vec \eta  = \frac{{\partial x}}{{\partial \eta }}{\text{d}}\eta \vec i + \frac{{\partial y}}{{\partial \eta }}{\text{d}}\eta \vec j + \frac{{\partial z}}{{\partial \eta }}{\text{d}}\eta \vec k

        {\text{d}}\vec \zeta  = \frac{{\partial x}}{{\partial \zeta }}{\text{d}}\zeta \vec i + \frac{{\partial y}}{{\partial \zeta }}{\text{d}}\zeta \vec j + \frac{{\partial z}}{{\partial \zeta }}{\text{d}}\zeta \vec k

    此时体积微元的变换可以表示为：

    .. math::
        {\text{d}}V = \left( {{\text{d}}\vec \xi  \times {\text{d}}\vec \eta } \right) \cdot {\text{d}}\vec \zeta  = \det \left( {\mathbf{J}} \right){\text{d}}\xi {\text{d}}\eta {\text{d}}\zeta

    由 :math:`{\text{d}}\vec \xi` 和 :math:`{\text{d}}\vec \eta` 组成的面积微元矢量 :math:`{\text{d}}\vec S` 可以表示为：

    .. math::
        {\text{d}}\vec S = {\text{d}}\vec \xi  \times {\text{d}}\vec \eta  = \det \left( {\begin{array}{*{20}{c}}
          {\vec i}&{\vec j}&{\vec k} \\
          {\frac{{\partial x}}{{\partial \xi }}}&{\frac{{\partial y}}{{\partial \xi }}}&{\frac{{\partial z}}{{\partial \xi }}} \\
          {\frac{{\partial x}}{{\partial \eta }}}&{\frac{{\partial y}}{{\partial \eta }}}&{\frac{{\partial z}}{{\partial \eta }}}
        \end{array}} \right){\text{d}}\xi {\text{d}}\eta

    此时面积微元的变换可以表示为：

    .. math::
        {\text{d}}S = \left\| {{\text{d}}\vec S} \right\| = \sqrt {\det {{\left( {\begin{array}{*{20}{c}}
          {\frac{{\partial x}}{{\partial \xi }}}&{\frac{{\partial y}}{{\partial \xi }}} \\
          {\frac{{\partial x}}{{\partial \eta }}}&{\frac{{\partial y}}{{\partial \eta }}}
        \end{array}} \right)}^2} + \det {{\left( {\begin{array}{*{20}{c}}
          {\frac{{\partial x}}{{\partial \xi }}}&{\frac{{\partial z}}{{\partial \xi }}} \\
          {\frac{{\partial x}}{{\partial \eta }}}&{\frac{{\partial z}}{{\partial \eta }}}
        \end{array}} \right)}^2} + \det {{\left( {\begin{array}{*{20}{c}}
          {\frac{{\partial y}}{{\partial \xi }}}&{\frac{{\partial z}}{{\partial \xi }}} \\
          {\frac{{\partial y}}{{\partial \eta }}}&{\frac{{\partial z}}{{\partial \eta }}}
        \end{array}} \right)}^2}} {\text{d}}\xi {\text{d}}\eta

    整理可得：

    .. math::
        {\text{d}}S = \sqrt {{{\left( {\frac{{\partial x}}{{\partial \xi }}\frac{{\partial y}}{{\partial \eta }} - \frac{{\partial x}}{{\partial \eta }}\frac{{\partial y}}{{\partial \xi }}} \right)}^2} + {{\left( {\frac{{\partial z}}{{\partial \xi }}\frac{{\partial x}}{{\partial \eta }} - \frac{{\partial z\partial x}}{{\partial \eta \partial \xi }}} \right)}^2} + {{\left( {\frac{{\partial y}}{{\partial \xi }}\frac{{\partial z}}{{\partial \eta }} - \frac{{\partial y}}{{\partial \eta }}\frac{{\partial z}}{{\partial \xi }}} \right)}^2}} {\text{d}}\xi {\text{d}}\eta

    同理:

    .. math::
        {\text{d}}S = \sqrt {{{\left( {\frac{{\partial x}}{{\partial \xi }}\frac{{\partial y}}{{\partial \zeta }} - \frac{{\partial x}}{{\partial \zeta }}\frac{{\partial y}}{{\partial \xi }}} \right)}^2} + {{\left( {\frac{{\partial z}}{{\partial \xi }}\frac{{\partial x}}{{\partial \zeta }} - \frac{{\partial z\partial x}}{{\partial \zeta \partial \xi }}} \right)}^2} + {{\left( {\frac{{\partial y}}{{\partial \xi }}\frac{{\partial z}}{{\partial \zeta }} - \frac{{\partial y}}{{\partial \zeta }}\frac{{\partial z}}{{\partial \xi }}} \right)}^2}} {\text{d}}\xi {\text{d}}\zeta

        {\text{d}}S = \sqrt {{{\left( {\frac{{\partial x}}{{\partial \zeta }}\frac{{\partial y}}{{\partial \eta }} - \frac{{\partial x}}{{\partial \eta }}\frac{{\partial y}}{{\partial \zeta }}} \right)}^2} + {{\left( {\frac{{\partial z}}{{\partial \zeta }}\frac{{\partial x}}{{\partial \eta }} - \frac{{\partial z\partial x}}{{\partial \eta \partial \zeta }}} \right)}^2} + {{\left( {\frac{{\partial y}}{{\partial \zeta }}\frac{{\partial z}}{{\partial \eta }} - \frac{{\partial y}}{{\partial \eta }}\frac{{\partial z}}{{\partial \zeta }}} \right)}^2}} {\text{d}}\zeta {\text{d}}\eta

    以八节点六面体等参元为例子::

                     (-1,1,1)        (1,1,1)
                      7---------------6
                     /|              /|
                    / |     ζ  η    / |
                   /  |     | /    /  |
        (-1,-1,1) 4---+-----|/----5 (1,-1,1)
                  |   |     o--ξ  |   |
                  |   3-----------+---2 (1,1,-1)
                  |  /(-1,1,-1)   |  /
                  | /             | /
                  |/              |/
                  0---------------1
                 (-1,-1,-1)      (1,-1,-1)

    边界上的第一类曲面积分可以表示为：

    .. math::
        {\left. {{{\mathbf{R}}^{\left( m \right)}}} \right|_{\xi  =  \pm 1}} = \int_{ - 1}^1 {\int_{ - 1}^1 {{{\left( {{{\mathbf{N}}^{\left( m \right)}}} \right)}^{\text{T}}}{\mathbf{\bar p}}} } \sqrt {{{\left( {\frac{{\partial x}}{{\partial \zeta }}\frac{{\partial y}}{{\partial \eta }} - \frac{{\partial x}}{{\partial \eta }}\frac{{\partial y}}{{\partial \zeta }}} \right)}^2} + {{\left( {\frac{{\partial z}}{{\partial \zeta }}\frac{{\partial x}}{{\partial \eta }} - \frac{{\partial z\partial x}}{{\partial \eta \partial \zeta }}} \right)}^2} + {{\left( {\frac{{\partial y}}{{\partial \zeta }}\frac{{\partial z}}{{\partial \eta }} - \frac{{\partial y}}{{\partial \eta }}\frac{{\partial z}}{{\partial \zeta }}} \right)}^2}} {\text{d}}\zeta {\text{d}}\eta

        {\left. {{{\mathbf{R}}^{\left( m \right)}}} \right|_{\eta  =  \pm 1}} = \int_{ - 1}^1 {\int_{ - 1}^1 {{{\left( {{{\mathbf{N}}^{\left( m \right)}}} \right)}^{\text{T}}}{\mathbf{\bar p}}} } \sqrt {{{\left( {\frac{{\partial x}}{{\partial \xi }}\frac{{\partial y}}{{\partial \zeta }} - \frac{{\partial x}}{{\partial \zeta }}\frac{{\partial y}}{{\partial \xi }}} \right)}^2} + {{\left( {\frac{{\partial z}}{{\partial \xi }}\frac{{\partial x}}{{\partial \zeta }} - \frac{{\partial z\partial x}}{{\partial \zeta \partial \xi }}} \right)}^2} + {{\left( {\frac{{\partial y}}{{\partial \xi }}\frac{{\partial z}}{{\partial \zeta }} - \frac{{\partial y}}{{\partial \zeta }}\frac{{\partial z}}{{\partial \xi }}} \right)}^2}} {\text{d}}\xi {\text{d}}\zeta

        {\left. {{{\mathbf{R}}^{\left( m \right)}}} \right|_{\zeta  =  \pm 1}} = \int_{ - 1}^1 {\int_{ - 1}^1 {{{\left( {{{\mathbf{N}}^{\left( m \right)}}} \right)}^{\text{T}}}{\mathbf{\bar p}}} } \sqrt {{{\left( {\frac{{\partial x}}{{\partial \xi }}\frac{{\partial y}}{{\partial \eta }} - \frac{{\partial x}}{{\partial \eta }}\frac{{\partial y}}{{\partial \xi }}} \right)}^2} + {{\left( {\frac{{\partial z}}{{\partial \xi }}\frac{{\partial x}}{{\partial \eta }} - \frac{{\partial z\partial x}}{{\partial \eta \partial \xi }}} \right)}^2} + {{\left( {\frac{{\partial y}}{{\partial \xi }}\frac{{\partial z}}{{\partial \eta }} - \frac{{\partial y}}{{\partial \eta }}\frac{{\partial z}}{{\partial \xi }}} \right)}^2}} {\text{d}}\xi {\text{d}}\eta

    通过选取合适的积分点，采用数值积分即可求得积分值。

    以四节点四面体等参元为例子::

       (0,0,1)
        3
        * **
        *   * *
        *     *  *
        *       *   2 (0,1,0)
        *        **  *
        ζ     *     * *
        |  η          **
        0---ξ * * * * * 1
       (0,0,0)         (1,0,0)

    边界上的第一类曲面积分可以表示为：

    .. math::
        {\left. {{{\mathbf{R}}^{\left( m \right)}}} \right|_{\xi  = 0}} = \int_0^1 {\int_0^1 {{{\left( {{{\mathbf{N}}^{\left( m \right)}}} \right)}^{\text{T}}}{\mathbf{\bar p}}} } \sqrt {{{\left( {\frac{{\partial x}}{{\partial \zeta }}\frac{{\partial y}}{{\partial \eta }} - \frac{{\partial x}}{{\partial \eta }}\frac{{\partial y}}{{\partial \zeta }}} \right)}^2} + {{\left( {\frac{{\partial z}}{{\partial \zeta }}\frac{{\partial x}}{{\partial \eta }} - \frac{{\partial z\partial x}}{{\partial \eta \partial \zeta }}} \right)}^2} + {{\left( {\frac{{\partial y}}{{\partial \zeta }}\frac{{\partial z}}{{\partial \eta }} - \frac{{\partial y}}{{\partial \eta }}\frac{{\partial z}}{{\partial \zeta }}} \right)}^2}} {\text{d}}\zeta {\text{d}}\eta

        {\left. {{{\mathbf{R}}^{\left( m \right)}}} \right|_{\eta  = 0}} = \int_0^1 {\int_0^1 {{{\left( {{{\mathbf{N}}^{\left( m \right)}}} \right)}^{\text{T}}}{\mathbf{\bar p}}} } \sqrt {{{\left( {\frac{{\partial x}}{{\partial \xi }}\frac{{\partial y}}{{\partial \zeta }} - \frac{{\partial x}}{{\partial \zeta }}\frac{{\partial y}}{{\partial \xi }}} \right)}^2} + {{\left( {\frac{{\partial z}}{{\partial \xi }}\frac{{\partial x}}{{\partial \zeta }} - \frac{{\partial z\partial x}}{{\partial \zeta \partial \xi }}} \right)}^2} + {{\left( {\frac{{\partial y}}{{\partial \xi }}\frac{{\partial z}}{{\partial \zeta }} - \frac{{\partial y}}{{\partial \zeta }}\frac{{\partial z}}{{\partial \xi }}} \right)}^2}} {\text{d}}\xi {\text{d}}\zeta

        {\left. {{{\mathbf{R}}^{\left( m \right)}}} \right|_{\zeta  = 0}} = \int_0^1 {\int_0^1 {{{\left( {{{\mathbf{N}}^{\left( m \right)}}} \right)}^{\text{T}}}{\mathbf{\bar p}}} } \sqrt {{{\left( {\frac{{\partial x}}{{\partial \xi }}\frac{{\partial y}}{{\partial \eta }} - \frac{{\partial x}}{{\partial \eta }}\frac{{\partial y}}{{\partial \xi }}} \right)}^2} + {{\left( {\frac{{\partial z}}{{\partial \xi }}\frac{{\partial x}}{{\partial \eta }} - \frac{{\partial z\partial x}}{{\partial \eta \partial \xi }}} \right)}^2} + {{\left( {\frac{{\partial y}}{{\partial \xi }}\frac{{\partial z}}{{\partial \eta }} - \frac{{\partial y}}{{\partial \eta }}\frac{{\partial z}}{{\partial \xi }}} \right)}^2}} {\text{d}}\xi {\text{d}}\eta

    对于斜面 1-2-3，我们有平面方程 :math:`\zeta  = 1 - \xi - \eta` ：

    .. math::
        {\left. {{{\mathbf{R}}^{\left( m \right)}}} \right|_{\zeta  = 1 - \xi  - \eta }} = \int_0^1 {\int_0^1 {{{\left( {{{\mathbf{N}}^{\left( m \right)}}} \right)}^{\text{T}}}{\mathbf{\bar p}}} } \sqrt {{{\left( {\frac{{\partial x}}{{\partial \xi }}\frac{{\partial y}}{{\partial \eta }} - \frac{{\partial x}}{{\partial \eta }}\frac{{\partial y}}{{\partial \xi }}} \right)}^2} + {{\left( {\frac{{\partial z}}{{\partial \xi }}\frac{{\partial x}}{{\partial \eta }} - \frac{{\partial z\partial x}}{{\partial \eta \partial \xi }}} \right)}^2} + {{\left( {\frac{{\partial y}}{{\partial \xi }}\frac{{\partial z}}{{\partial \eta }} - \frac{{\partial y}}{{\partial \eta }}\frac{{\partial z}}{{\partial \xi }}} \right)}^2}} \sqrt {1 + {{\left( {\frac{{\partial \zeta }}{{\partial \xi }}} \right)}^2} + {{\left( {\frac{{\partial \zeta }}{{\partial \eta }}} \right)}^2}} {\text{d}}\xi {\text{d}}\eta

    所以有：

    .. math::
        {\left. {{{\mathbf{R}}^{\left( m \right)}}} \right|_{\zeta  = 1 - \xi  - \eta }} = \int_0^1 {\int_0^1 {{{\left( {{{\mathbf{N}}^{\left( m \right)}}} \right)}^{\text{T}}}{\mathbf{\bar p}}} } \sqrt {{{\left( {\frac{{\partial x}}{{\partial \xi }}\frac{{\partial y}}{{\partial \eta }} - \frac{{\partial x}}{{\partial \eta }}\frac{{\partial y}}{{\partial \xi }}} \right)}^2} + {{\left( {\frac{{\partial z}}{{\partial \xi }}\frac{{\partial x}}{{\partial \eta }} - \frac{{\partial z\partial x}}{{\partial \eta \partial \xi }}} \right)}^2} + {{\left( {\frac{{\partial y}}{{\partial \xi }}\frac{{\partial z}}{{\partial \eta }} - \frac{{\partial y}}{{\partial \eta }}\frac{{\partial z}}{{\partial \xi }}} \right)}^2}} \sqrt 3 {\text{d}}\xi {\text{d}}\eta

    通过选取合适的积分点，采用数值积分即可求得积分值。
    """

    __slots__ = BaseBC.__slots__ + []

    def __init__(self, bc: BC, dof: Dof, mesh_data: MeshData, solver: Solver, amplitude: Optional[Amplitude]) -> None:
        super().__init__(bc, dof, mesh_data, solver, amplitude)
        self.create_dof_values()

    def get_surface_from_bc_element(self, bc_element_id: int, bc_element: ndarray) -> list[tuple[int, str]]:
        nodes = self.mesh_data.nodes
        elements = self.mesh_data.elements
        element_surface = []
        for element_id, element in enumerate(elements):
            is_element_surface = all(in1d(bc_element, element))
            if is_element_surface:
                nodes_in_element = in1d(element, bc_element)
                connectivity = elements[element_id]
                node_coords = nodes[connectivity]
                iso_element_type = get_iso_element_type(node_coords)
                iso_element_shape = iso_element_shape_dict[iso_element_type]
                surface_names = [surface_name for surface_name, nodes_on_surface in
                                 iso_element_shape.nodes_on_surface_dict.items() if
                                 all(nodes_on_surface == nodes_in_element)]
                if len(surface_names) == 1:
                    element_surface.append((element_id, surface_names[0]))
                else:
                    raise NotImplementedError(error_style(f'the surface of element {element_id} is wrong'))

        if len(element_surface) == 1:
            return element_surface
        else:
            raise NotImplementedError(error_style(f'the surface of bc_element {bc_element_id} is wrong'))

    def get_surface_from_elements_nodes(self, element_id: int, node_ids: list[int]) -> list[tuple[int, str]]:
        nodes = self.mesh_data.nodes
        elements = self.mesh_data.elements
        element_surface = []
        nodes_in_element = in1d(elements[element_id], node_ids)
        connectivity = elements[element_id]
        node_coords = nodes[connectivity]
        iso_element_type = get_iso_element_type(node_coords)
        iso_element_shape = iso_element_shape_dict[iso_element_type]
        surface_names = [surface_name for surface_name, nodes_on_surface in
                         iso_element_shape.nodes_on_surface_dict.items() if
                         sum(logical_and(nodes_in_element, nodes_on_surface)) == len(
                             iso_element_shape.bc_surface_nodes_dict[surface_name])]

        for surface_name in surface_names:
            element_surface.append((element_id, surface_name))

        if 1 <= len(element_surface) <= iso_element_shape.bc_surface_number:
            return element_surface
        else:
            raise NotImplementedError(error_style(f'the surface of element {element_id} is wrong'))

    def create_dof_values(self) -> None:
        dimension = self.mesh_data.dimension
        nodes = self.mesh_data.nodes
        elements = self.mesh_data.elements
        bc_elements = self.mesh_data.bc_elements

        node_sets = self.bc.node_sets
        element_sets = self.bc.element_sets
        bc_element_sets = self.bc.bc_element_sets
        bc_value = self.bc.value

        if bc_element_sets is not None:
            bc_element_ids = []
            for bc_element_set in bc_element_sets:
                bc_element_ids += list(self.mesh_data.bc_element_sets[bc_element_set])
            for bc_element_id in set(bc_element_ids):
                self.bc_surface += self.get_surface_from_bc_element(bc_element_id, bc_elements[bc_element_id])
        elif element_sets is not None and node_sets is not None:
            if element_sets == node_sets:
                element_ids = []
                for element_set in element_sets:
                    element_ids += list(self.mesh_data.element_sets[element_set])
                node_ids = []
                for node_set in node_sets:
                    node_ids += list(self.mesh_data.node_sets[node_set])
                for element_id in set(element_ids):
                    self.bc_surface += self.get_surface_from_elements_nodes(element_id, node_ids)
            else:
                raise NotImplementedError(
                    error_style(f'the name of element_sets {element_sets} and node_sets {node_sets} must be the same'))

        dof_ids = []
        bc_fext = []
        bc_dof_names = self.bc.dof
        dof_names = self.dof.names

        for element_id, surface_name in self.bc_surface:
            connectivity = elements[element_id]
            node_coords = nodes[connectivity]
            iso_element_type = get_iso_element_type(node_coords)
            iso_element_shape = iso_element_shape_dict[iso_element_type]

            nodes_number = iso_element_shape.nodes_number
            bc_qp_weights = iso_element_shape.bc_qp_weights
            bc_qp_number = len(bc_qp_weights)
            bc_qp_shape_values = iso_element_shape.bc_qp_shape_values_dict[surface_name]
            bc_qp_shape_gradients = iso_element_shape.bc_qp_shape_gradients_dict[surface_name]
            bc_surface_coord = iso_element_shape.bc_surface_coord_dict[surface_name]
            surface_local_nodes = array(iso_element_shape.bc_surface_nodes_dict[surface_name])
            surface_nodes = elements[element_id][surface_local_nodes]
            surface_dof_ids = []
            for node_index in surface_nodes:
                for _, bc_dof_name in enumerate(bc_dof_names):
                    surface_dof_id = node_index * len(dof_names) + dof_names.index(bc_dof_name)
                    surface_dof_ids.append(surface_dof_id)

            dof_ids += surface_dof_ids

            element_fext = zeros(nodes_number)

            for i in range(bc_qp_number):
                bc_qp_jacobi = dot(bc_qp_shape_gradients[i], node_coords).transpose()
                bc_qp_jacobi_sub = delete(bc_qp_jacobi, bc_surface_coord[0], axis=1)
                surface_weight = bc_surface_coord[3]
                if dimension == 2:
                    s = sum(bc_qp_jacobi_sub ** 2)
                    qp_fext = bc_qp_shape_values[i].transpose() * bc_qp_weights[i] * bc_value * sqrt(s) * surface_weight
                elif dimension == 3:
                    s = 0
                    for row in range(bc_qp_jacobi_sub.shape[0]):
                        s += det(delete(bc_qp_jacobi_sub, row, axis=0)) ** 2
                    qp_fext = bc_qp_shape_values[i].transpose() * bc_qp_weights[i] * bc_value * sqrt(s) * surface_weight
                else:
                    raise NotImplementedError(error_style(f'dimension {dimension} is not supported of the Neumann boundary condition'))

                element_fext += qp_fext

            surface_fext = []
            for fext in element_fext[surface_local_nodes]:
                for _ in range(len(bc_dof_names)):
                    surface_fext.append(fext)

            bc_fext += list(surface_fext)

        self.dof_ids = array(dof_ids)
        self.bc_fext = array(bc_fext)


if __name__ == "__main__":
    from pyfem.io.Properties import Properties

    # props = Properties()
    # props.read_file(r'..\..\..\examples\mechanical\plane\Job-1.toml')
    # bc_data = NeumannBCDistributed(props.bcs[2], props.dof, props.mesh_data, props.solver, None)
    # bc_data.show()

    # props = Properties()
    # props.read_file(r'..\..\..\examples\mechanical\quad8\Job-1.toml')
    # bc_data = NeumannBCDistributed(props.bcs[2], props.dof, props.mesh_data, props.solver, props.amplitudes[0])
    # bc_data.show()

    props = Properties()
    props.read_file(r'..\..\..\examples\mechanical\1element\tetra4\Job-1.toml')
    bc_data = NeumannBCDistributed(props.bcs[3], props.dof, props.mesh_data, props.solver, None)
    bc_data.show()
