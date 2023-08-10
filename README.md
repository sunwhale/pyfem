# pyfem

pyfem是一个完全基于python语言实现的极简有限元求解器。依赖的第三方库包括numpy、scipy和meshio等，主要用于有限元方法的学习、有限元算法验证和快速建立材料本构模型的程序原型。


[![Codacy Badge](https://app.codacy.com/project/badge/Grade/ab5bca55d85d45d4aa4336ccae058316)](https://app.codacy.com/gh/sunwhale/pyfem/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)

## Installation 安装

Use the package manager [pip](https://pypi.org/project/pyfem/) to install pyfem. 使用pip命令安装。

```bash
pip install pyfem
```

## Tutorial 指南

### Run in command line 在命令行运行:

```bash
pyfem --help
```

### Run the first example 执行第一个算例:

当前算例文件存储目录 examples\tutorial，该算例定义了一个二维平面应变模型，材料为塑性随动强化，载荷为y方向的循环拉伸-压缩。

```bash
cd examples\tutorial
pyfem -i Job-1.toml
```

#### 算例配置文件 Job-1.toml:

```toml
title = "Job-1"

[mesh] # 前处理网格文件
type = "gmsh"
file = 'mesh.msh'

[dof] # 自由度
names = ["u1", "u2"]
order = 1
family = "LAGRANGE"

[[amplitudes]] # 幅值列表
name = "Amp-1"
type = "TabularAmplitude"
start = 0.0
data = [
    [0.0, 0.0],
    [0.5, 1.0],
    [1.0, 0.0],
    [1.5, -1.0],
    [2.0, 0.0],
    [2.5, 1.0],
    [3.0, 0.0],
    [3.5, -1.0],
    [4.0, 0.0],
    [4.5, 1.0],
    [5.0, 0.0],
]

[[bcs]] # 边界条件列表
name = "BC-1"
category = "DirichletBC"
type = ""
dof = ["u2"]
node_sets = ['bottom']
element_sets = []
value = 0.0

[[bcs]] # 边界条件列表
name = "BC-2"
category = "DirichletBC"
type = ""
dof = ["u1"]
node_sets = ['left']
element_sets = []
value = 0.0

[[bcs]] # 边界条件列表
name = "BC-3"
category = "DirichletBC"
type = ""
dof = ["u2"]
node_sets = ['top']
element_sets = []
value = 0.01
amplitude_name = "Amp-1"

[solver] # 求解器属性
type = "NonlinearSolver"
option = "NewtonRaphson"
total_time = 5.0
start_time = 0.0
max_increment = 1000000
initial_dtime = 0.05
max_dtime = 0.05
min_dtime = 0.001

[[materials]] # 材料列表
name = "Material-1"
category = "Plastic"
type = "KinematicHardening"
data = [100000.0, 0.25, 400.0, 1000.0]

[[sections]] # 截面列表
name = "Section-1"
category = "Solid"
type = "PlaneStrain"
option = "SmallStrain"
element_sets = ["rectangle"]
material_names = ["Material-1"]
data = []

[[outputs]] # 输出列表
type = "vtk"
field_outputs = ['S11', 'S22', 'S12', 'E11', 'E22', 'E12']
on_screen = false
```

#### 采用gmsh格式的网格文件 mesh.msh:

```
$MeshFormat
4.1 0 8
$EndMeshFormat
$PhysicalNames
5
1 5 "left"
1 6 "right"
1 7 "top"
1 8 "bottom"
2 9 "rectangle"
$EndPhysicalNames
$Entities
4 4 1 0
1 0 0 0 0 
2 1 0 0 0 
3 1 1 0 0 
4 0 1 0 0 
1 0 0 0 1 0 0 1 8 2 1 -2 
2 1 0 0 1 1 0 1 6 2 2 -3 
3 0 1 0 1 1 0 1 7 2 3 -4 
4 0 0 0 0 1 0 1 5 2 4 -1 
1 0 0 0 1 1 0 1 9 4 3 4 1 2 
$EndEntities
$Nodes
9 9 1 9
0 1 0 1
1
0 0 0
0 2 0 1
2
1 -0 0
0 3 0 1
3
1 1 0
0 4 0 1
4
0 1 0
1 1 0 1
5
0.4999999999986921 0 0
1 2 0 1
6
1 0.4999999999986921 0
1 3 0 1
7
0.5000000000020595 1 0
1 4 0 1
8
0 0.5000000000020595 0
2 1 0 1
9
0.5000000000003758 0.5000000000003758 0
$EndNodes
$Elements
5 12 1 12
1 1 1 2
1 1 5 
2 5 2 
1 2 1 2
3 2 6 
4 6 3 
1 3 1 2
5 3 7 
6 7 4 
1 4 1 2
7 4 8 
8 8 1 
2 1 3 4
9 3 7 9 6 
10 6 9 5 2 
11 7 4 8 9 
12 9 8 1 5 
$EndElements
```

### Documents 帮助文档
[https://pyfem-doc.readthedocs.io/](https://pyfem-doc.readthedocs.io/)

## Development

### ToDo list

- [x] 增加Neumann边界条件，支持concentrated force，distributed和pressure定义方式
- [ ] 增加hdf5计算结果输出格式
- [ ] 完善帮助文档
- [ ] 完善输入文件的校检
- [x] 增加测试模块
- [ ] 增加日志模块
- [ ] 增加后台运行模式
- [ ] 处理平面应力状态的面外应力平衡
- [x] 增加粘弹性力学本构模型
- [ ] 增加晶体塑性力学本构模型
- [x] 增加温度场求解单元
- [x] 增加温度场-位移场耦合求解单元
- [x] 增加相场-位移场耦合求解单元
- [ ] 增加内聚区单元
- [ ] 增加动力学求解器
- [ ] 建立前处理界面

### Bug list

- [ ] 采用abaqus网格文件时，如果存在node不属于任何element则在计算时会导致全局刚度矩阵奇异。
