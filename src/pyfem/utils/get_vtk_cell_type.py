# -*- coding: utf-8 -*-
"""

"""


def get_vtk_cell_type(dimension, nodes_number):
    """
    根据维度和节点数判断VTK单元格类型

    参数:
    dimension: 单元格维度 (0, 1, 2, 3)
    nodes_number: 单元格的节点数

    返回:
    cell_type_name: 单元格类型名称
    cell_type_int: 单元格类型整数值
    """

    # 常见单元格类型
    common_cells = {
        # 0D元素
        (0, 1): ("VTK_VERTEX", 1),
        (0, 0): ("VTK_EMPTY_CELL", 0),

        # 1D元素
        (1, 2): ("VTK_LINE", 3),  # 线性线段
        (1, 4): ("VTK_CUBIC_LINE", 35),  # 三次曲线

        # 2D元素 - 三角形
        (2, 3): ("VTK_TRIANGLE", 5),  # 线性三角形
        (2, 6): ("VTK_QUADRATIC_TRIANGLE", 22),  # 二次三角形

        # 2D元素 - 四边形
        (2, 4): ("VTK_QUAD", 9),  # 线性四边形
        (2, 8): ("VTK_QUADRATIC_QUAD", 23),  # 二次四边形
        (2, 9): ("VTK_BIQUADRATIC_QUAD", 28),  # 双二次四边形

        # 2D元素 - 特殊
        (2, 0): ("VTK_EMPTY_CELL", 0),

        # 3D元素 - 四面体
        (3, 4): ("VTK_TETRA", 10),  # 线性四面体
        (3, 10): ("VTK_QUADRATIC_TETRA", 24),  # 二次四面体

        # 3D元素 - 六面体
        (3, 8): ("VTK_HEXAHEDRON", 12),  # 线性六面体
        (3, 20): ("VTK_QUADRATIC_HEXAHEDRON", 12),  # 二次六面体
        # (3, 20): ("VTK_QUADRATIC_HEXAHEDRON", 25),  # 二次六面体

        # 3D元素 - 棱柱
        (3, 6): ("VTK_WEDGE", 13),  # 三棱柱
        (3, 15): ("VTK_QUADRATIC_WEDGE", 26),  # 二次三棱柱

        # 3D元素 - 金字塔
        (3, 5): ("VTK_PYRAMID", 14),  # 金字塔
        (3, 13): ("VTK_QUADRATIC_PYRAMID", 27),  # 二次金字塔
    }

    # 首先检查常见类型
    key = (dimension, nodes_number)
    if key in common_cells:
        return common_cells[key]

    return "UNKNOWN", -1


# VTK单元格类型完整字典
vtk_cell_types = {
    # ==================== 0D元素 ====================
    "VTK_EMPTY_CELL": 0,  # 空单元格，无几何结构
    "VTK_VERTEX": 1,  # 单个顶点，0D单元
    "VTK_POLY_VERTEX": 2,  # 多顶点，一系列不连接的顶点

    # ==================== 1D元素 ====================
    "VTK_LINE": 3,  # 线性线段，2个节点
    "VTK_POLY_LINE": 4,  # 多段线，一系列连接的线段
    "VTK_CUBIC_LINE": 35,  # 三次曲线线段，4个节点

    # ==================== 2D元素 ====================
    # --- 三角形类 ---
    "VTK_TRIANGLE": 5,  # 线性三角形，3个节点
    "VTK_TRIANGLE_STRIP": 6,  # 三角形条带，共享边的三角形序列
    "VTK_QUADRATIC_TRIANGLE": 22,  # 二次三角形，6个节点(3个顶点+3个边中点)
    "VTK_BIQUADRATIC_TRIANGLE": 34,  # 双二次三角形，更多节点用于更高阶插值

    # --- 四边形类 ---
    "VTK_PIXEL": 8,  # 像素单元，类似四边形但边与坐标轴平行
    "VTK_QUAD": 9,  # 线性四边形，4个节点
    "VTK_QUADRATIC_QUAD": 23,  # 二次四边形，8个节点(4个顶点+4个边中点)
    "VTK_BIQUADRATIC_QUAD": 28,  # 双二次四边形，9个节点

    # --- 多边形类 ---
    "VTK_POLYGON": 7,  # 多边形，任意数量的顶点
    "VTK_PENTAGONAL_PRISM": 15,  # 五边形棱柱(虽然是3D，但由2D五边形拉伸得到)
    "VTK_HEXAGONAL_PRISM": 16,  # 六边形棱柱

    # --- 高阶2D单元 ---
    "VTK_QUADRATIC_LINEAR_QUAD": 30,  # 二次线性混合四边形

    # ==================== 3D元素 ====================
    # --- 四面体类 ---
    "VTK_TETRA": 10,  # 线性四面体，4个节点
    "VTK_QUADRATIC_TETRA": 24,  # 二次四面体，10个节点

    # --- 六面体类 ---
    "VTK_VOXEL": 11,  # 体素，类似六面体但方向与坐标轴平行，8个节点
    "VTK_HEXAHEDRON": 12,  # 线性六面体，8个节点
    "VTK_QUADRATIC_HEXAHEDRON": 25,  # 二次六面体，20个节点

    # --- 棱柱类 ---
    "VTK_WEDGE": 13,  # 三棱柱，6个节点
    "VTK_QUADRATIC_WEDGE": 26,  # 二次三棱柱，15个节点

    # --- 金字塔类 ---
    "VTK_PYRAMID": 14,  # 金字塔形(四边形底面+一个顶点)，5个节点
    "VTK_QUADRATIC_PYRAMID": 27,  # 二次金字塔，13个节点

    # --- 其他3D单元 ---
    "VTK_CONVEX_POINT_SET": 41,  # 凸点集，任意凸多面体
    "VTK_POLYHEDRON": 42,  # 多面体，任意非凸多面体

    # ==================== 高阶/参数化单元 ====================
    "VTK_PARAMETRIC_CURVE": 51,
    "VTK_PARAMETRIC_SURFACE": 52,
    "VTK_PARAMETRIC_TRI_SURFACE": 53,
    "VTK_PARAMETRIC_QUAD_SURFACE": 54,
    "VTK_PARAMETRIC_TETRA_REGION": 55,
    "VTK_PARAMETRIC_HEX_REGION": 56,

    # ==================== 拉格朗日单元 ====================
    "VTK_LAGRANGE_CURVE": 68,
    "VTK_LAGRANGE_TRIANGLE": 69,
    "VTK_LAGRANGE_QUADRILATERAL": 70,
    "VTK_LAGRANGE_TETRAHEDRON": 71,
    "VTK_LAGRANGE_HEXAHEDRON": 72,
    "VTK_LAGRANGE_WEDGE": 73,
    "VTK_LAGRANGE_PYRAMID": 74,

    # ==================== 贝塞尔单元 ====================
    "VTK_BEZIER_CURVE": 75,
    "VTK_BEZIER_TRIANGLE": 76,
    "VTK_BEZIER_QUADRILATERAL": 77,
    "VTK_BEZIER_TETRAHEDRON": 78,
    "VTK_BEZIER_HEXAHEDRON": 79,
    "VTK_BEZIER_WEDGE": 80,
    "VTK_BEZIER_PYRAMID": 81,
}

# 反向字典：整数值 -> 类型名
vtk_cell_types_reverse = {v: k for k, v in vtk_cell_types.items()}

if __name__ == "__main__":
    print("VTK单元格类型判断示例")
    print("=" * 60)

    # 根据维度和节点数判断单元格类型
    test_cases = [
        (0, 1, "单个顶点"),
        (1, 2, "线性线段"),
        (2, 3, "三角形"),
        (2, 4, "四边形"),
        (2, 6, "二次三角形"),
        (3, 4, "四面体"),
        (3, 5, "金字塔"),
        (3, 6, "三棱柱"),
        (3, 8, "六面体"),
        (3, 10, "二次四面体"),
        (3, 20, "二次六面体"),
    ]

    print("根据维度和节点数判断单元格类型:")
    for dim, nodes, desc in test_cases:
        name, value = get_vtk_cell_type(dim, nodes)
        print(f"维度{dim}, 节点数{nodes:2d} -> {name:25} (值:{value:3d}) : {desc}")

    print("\n" + "=" * 60)
