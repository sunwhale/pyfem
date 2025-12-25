"""
测试DistributedLoad类的程序
演示如何在有限元分析中施加分布载荷
"""

import numpy as np
from pyfem.util.shapeFunctions import getElemShapeData


# 由于原代码在包内，我们创建一个简化的测试版本
class MockElement:
    """模拟Element基类"""

    def __init__(self, elnodes, props):
        self.elnodes = elnodes
        self.props = props
        self.rank = props.get('rank', 2)
        self.trac = props.get('trac', None)
        self.pressure = props.get('pressure', None)

        # 模拟solverStat对象
        class MockSolverStat:
            def __init__(self):
                self.lam = 1.0  # 载荷因子

        self.solverStat = MockSolverStat()


class DistributedLoad(MockElement):
    """简化的DistributedLoad类用于测试"""

    def __init__(self, elnodes, props):
        MockElement.__init__(self, elnodes, props)

        if self.rank == 2:
            self.dofTypes = ['u', 'v']
        elif self.rank == 3:
            self.dofTypes = ['u', 'v', 'w']

        if self.trac is not None:
            self.trac = np.array(self.trac)

    def loadFactor(self):
        """模拟载荷因子方法"""
        return 1.0

    def getExternalForce(self, elemdat):
        """计算外部力"""
        sData = self.getShapeData(elemdat)

        # 初始化节点力向量
        nNodes = elemdat.coords.shape[0]
        elemdat.fint = np.zeros(self.rank * nNodes)

        for iData in sData:
            N = self.getNmatrix(iData.h)
            trac = self.getTraction(iData.normal)

            # 计算节点力贡献
            elemdat.fint += np.dot(trac, N) * iData.weight

        # 乘以载荷因子
        elemdat.fint *= self.loadFactor()
        return elemdat.fint

    def getNmatrix(self, h):
        """构造形函数矩阵"""
        N = np.zeros((self.rank, self.rank * len(h)))

        for i, a in enumerate(h):
            for j in range(self.rank):
                N[j, self.rank * i + j] = a

        return N

    def getShapeData(self, elemdat):
        """获取形状数据"""
        crd = elemdat.coords
        nNod = crd.shape[0]

        if self.rank == 2:
            b = np.zeros(3)
            b[2] = 1.0

            if nNod == 2:
                sData = getElemShapeData(elemdat.coords, elemType="Line2")
                a = self.getDirection(crd, 1, 0)
            elif nNod == 3:
                sData = getElemShapeData(elemdat.coords, elemType="Line3")
                a = self.getDirection(crd, 2, 0)
            else:
                raise RuntimeError("The rank is 2, the number of nodes must be 2 or 3.")
        elif self.rank == 3:
            if nNod == 3:
                # 三角形单元
                sData = getElemShapeData(elemdat.coords, elemType="Tria3")
                a = self.getDirection(crd, 1, 0)
                b = self.getDirection(crd, 2, 0)
            elif nNod == 4:
                # 四边形单元
                sData = getElemShapeData(elemdat.coords, elemType="Quad4")
                a = self.getDirection(crd, 1, 0)
                b = self.getDirection(crd, 2, 0)
            else:
                raise RuntimeError("The rank is 3, unsupported number of nodes.")
        else:
            raise RuntimeError("The element must be rank 2 or 3.")

        # 计算法向量
        for iData in sData:
            iData.normal = np.cross(a, b)
            iData.normal *= 1.0 / np.sqrt(np.dot(iData.normal, iData.normal))

        return sData

    def getTraction(self, normal):
        """计算牵引力"""
        if hasattr(self, "pressure") and self.pressure is not None:
            return self.solverStat.lam * normal * self.pressure
        elif hasattr(self, "trac") and self.trac is not None:
            return self.solverStat.lam * self.trac
        else:
            raise RuntimeError("Define either pressure or trac")

    def getDirection(self, crd, i, j):
        """计算方向向量"""
        direc = np.zeros(3)
        rank = crd.shape[1]
        direc[:rank] = crd[i, :] - crd[j, :]
        return direc


def test_2d_line_load():
    """测试2D线分布载荷"""
    print("=" * 60)
    print("测试1: 2D线分布载荷")
    print("=" * 60)

    # 定义2节点线单元
    nodes = [1, 2]

    # 情况1: 指定方向的分布力
    props = {
        'rank': 2,
        'trac': [10.0, 0.0]  # x方向10 N/m的分布力
    }

    # 创建单元数据
    class ElemDat:
        def __init__(self, coords):
            self.coords = coords
            self.fint = None

    # 单元坐标 (水平线，长1m)
    coords = np.array([[0.0, 0.0],
                       [1.0, 0.0]])

    elemdat = ElemDat(coords)

    # 创建分布载荷单元
    load_elem = DistributedLoad(nodes, props)

    # 计算等效节点力
    fint = load_elem.getExternalForce(elemdat)

    print(f"单元类型: 2节点线单元")
    print(f"节点坐标: \n{coords}")
    print(f"分布载荷: {props['trac']} N/m")
    print(f"等效节点力: {fint}")
    print(f"理论值: 每个节点x方向5N, y方向0N")
    print()

    # 情况2: 压力载荷（垂直于表面）
    props2 = {
        'rank': 2,
        'pressure': 5.0  # 5 N/m²的压力
    }

    # 创建斜线单元（45度角）
    coords2 = np.array([[0.0, 0.0],
                        [1.0, 1.0]])

    elemdat2 = ElemDat(coords2)
    load_elem2 = DistributedLoad(nodes, props2)
    fint2 = load_elem2.getExternalForce(elemdat2)

    print(f"单元类型: 2节点线单元（45度斜线）")
    print(f"节点坐标: \n{coords2}")
    print(f"压力载荷: {props2['pressure']} N/m²")
    print(f"等效节点力: {fint2}")
    print()


def test_3d_surface_load():
    """测试3D表面分布载荷"""
    print("=" * 60)
    print("测试2: 3D表面分布载荷")
    print("=" * 60)

    # 定义4节点四边形单元
    nodes = [1, 2, 3, 4]

    # 情况1: 指定方向的牵引力
    props = {
        'rank': 3,
        'trac': [0.0, 0.0, -10.0]  # z方向-10 N/m²的分布力
    }

    class ElemDat:
        def __init__(self, coords):
            self.coords = coords
            self.fint = None

    # 创建位于xy平面的正方形单元（1m×1m）
    coords = np.array([
        [0.0, 0.0, 0.0],  # 节点1
        [1.0, 0.0, 0.0],  # 节点2
        [1.0, 1.0, 0.0],  # 节点3
        [0.0, 1.0, 0.0]  # 节点4
    ])

    elemdat = ElemDat(coords)

    # 创建分布载荷单元
    load_elem = DistributedLoad(nodes, props)

    # 计算等效节点力
    fint = load_elem.getExternalForce(elemdat)

    print(f"单元类型: 4节点四边形单元")
    print(f"节点坐标: \n{coords}")
    print(f"分布载荷: {props['trac']} N/m²")
    print(f"等效节点力: \n{fint.reshape(-1, 3)}")
    print(f"节点力总和: {np.sum(fint.reshape(-1, 3), axis=0)}")
    print()

    # 情况2: 三角形单元压力载荷
    print("测试3: 3D三角形单元压力载荷")
    print("-" * 40)

    nodes_tria = [1, 2, 3]
    props_tria = {
        'rank': 3,
        'pressure': 20.0  # 20 N/m²的压力
    }

    # 创建三角形单元（在xy平面）
    coords_tria = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ])

    elemdat_tria = ElemDat(coords_tria)
    load_elem_tria = DistributedLoad(nodes_tria, props_tria)
    fint_tria = load_elem_tria.getExternalForce(elemdat_tria)

    print(f"单元类型: 3节点三角形单元")
    print(f"节点坐标: \n{coords_tria}")
    print(f"压力载荷: {props_tria['pressure']} N/m²")
    print(f"等效节点力: \n{fint_tria.reshape(-1, 3)}")
    print(f"面积: 0.5 m²")
    print(f"总力: {20.0 * 0.5} N (z方向)")
    print()


def test_verification():
    """验证测试"""
    print("=" * 60)
    print("验证测试")
    print("=" * 60)

    # 测试1: 2D线单元分布力验证
    print("验证1: 2D线单元等效节点力")
    print("-" * 40)

    # 简单解析解：均匀分布力q，长度L，总力F=q*L
    # 等效节点力：每个节点F/2
    q = 10.0  # N/m
    L = 1.0  # m
    F_total = q * L
    expected_node_force = F_total / 2

    nodes = [1, 2]
    props = {'rank': 2, 'trac': [q, 0.0]}

    class ElemDat:
        def __init__(self, coords):
            self.coords = coords
            self.fint = None

    coords = np.array([[0.0, 0.0], [L, 0.0]])
    elemdat = ElemDat(coords)
    load_elem = DistributedLoad(nodes, props)
    fint = load_elem.getExternalForce(elemdat)

    print(f"理论总力: {F_total} N")
    print(f"理论节点力: [{expected_node_force}, 0] N")
    print(f"计算节点力: {fint}")
    print(f"误差: {np.abs(fint[0] - expected_node_force)}")

    # 测试2: 合力为零检查（封闭单元）
    print("\n验证2: 封闭单元合力检查")
    print("-" * 40)

    # 创建一个正方形环（4个线单元）
    square_nodes = [
        [0.0, 0.0],  # 节点1
        [1.0, 0.0],  # 节点2
        [1.0, 1.0],  # 节点3
        [0.0, 1.0]  # 节点4
    ]

    # 对整个环施加均匀压力
    total_force = np.zeros(2)

    # 创建4个边单元
    edges = [
        ([1, 2], np.array([square_nodes[0], square_nodes[1]])),  # 底边
        ([2, 3], np.array([square_nodes[1], square_nodes[2]])),  # 右边
        ([3, 4], np.array([square_nodes[2], square_nodes[3]])),  # 顶边
        ([4, 1], np.array([square_nodes[3], square_nodes[0]]))  # 左边
    ]

    for i, (edge_nodes, edge_coords) in enumerate(edges):
        props_edge = {'rank': 2, 'pressure': 5.0}
        elemdat_edge = ElemDat(edge_coords)
        load_elem_edge = DistributedLoad(edge_nodes, props_edge)
        fint_edge = load_elem_edge.getExternalForce(elemdat_edge)

        # 累加节点力
        # 注意：每个节点被两个单元共享，实际应用中需要组装全局力向量
        print(f"边{i + 1}节点力: {fint_edge}")
        total_force += fint_edge

    print(f"总合力: {total_force}")
    print("理论上合力应为零（封闭区域）")


def main():
    """主测试函数"""
    print("分布式载荷单元测试程序")
    print("=" * 60)

    # 运行测试
    test_2d_line_load()
    test_3d_surface_load()
    test_verification()

    print("=" * 60)
    print("测试完成！")


if __name__ == "__main__":
    # 注意：由于getElemShapeData需要实际的pyfem库，
    # 这里我们创建一个简单的模拟版本

    class ShapeData:
        """模拟形状数据类"""

        def __init__(self, h, weight):
            self.h = h
            self.weight = weight
            self.normal = None


    def getElemShapeData(coords, elemType):
        """模拟获取单元形状数据"""
        if elemType == "Line2":
            # 2节点线单元：1个高斯点
            return [ShapeData(h=np.array([0.5, 0.5]), weight=1.0)]
        elif elemType == "Line3":
            # 3节点线单元：2个高斯点
            return [
                ShapeData(h=np.array([0.5, 0.5, 0.0]), weight=0.5),
                ShapeData(h=np.array([0.0, 0.5, 0.5]), weight=0.5)
            ]
        elif elemType == "Quad4":
            # 4节点四边形单元：1个高斯点（简化）
            return [ShapeData(h=np.array([0.25, 0.25, 0.25, 0.25]), weight=1.0)]
        elif elemType == "Tria3":
            # 3节点三角形单元：1个高斯点（简化）
            return [ShapeData(h=np.array([1 / 3, 1 / 3, 1 / 3]), weight=0.5)]
        else:
            raise ValueError(f"不支持的单元类型: {elemType}")


    # 将模拟函数添加到模块中
    import sys

    sys.modules[__name__].getElemShapeData = getElemShapeData

    # 运行测试
    main()