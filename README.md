# pyfem

A finite element package in python.

[![Codacy Badge](https://app.codacy.com/project/badge/Grade/ab5bca55d85d45d4aa4336ccae058316)](https://app.codacy.com/gh/sunwhale/pyfem/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)

## Installation

Use the package manager [pip](https://pypi.org/project/pyfem/) to install pyfem.

```bash
pip install pyfem
```

## Development

### ToDo list

- [ ] 增加Neumann边界条件，支持concentrated force，pressure，traction和函数定义方式
- [ ] 增加hdf5计算结果输出格式
- [ ] 完善帮助文档
- [ ] 完善输入文件的校检
- [ ] 增加测试模块
- [ ] 增加日志模块
- [ ] 增加后台运行模式
- [ ] 处理平面应力状态的面外应力平衡
- [x] 增加粘弹性力学本构模型
- [ ] 增加晶体塑性力学本构模型
- [ ] 增加温度场求解单元
- [ ] 增加温度场-位移场耦合求解单元
- [ ] 增加相场-位移场耦合求解单元
- [ ] 增加内聚区单元
- [ ] 增加动力学求解器
- [ ] 建立前处理界面

## Tutorial

#### Run in command line:

```bash
pyfem -i Job-1.toml
```

#### Example with geometry file:

```toml
title = "Job-1"

[mesh]
type = "abaqus"
file = "Job-1.inp"

[dof]
names = ["u1", "u2", "u3"]
order = 1
family = "LAGRANGE"

[[bcs]]
name = "BC-1"
category = "DirichletBC"
type = ""
dof = ["u1"]
node_sets = ['Set-X0']
element_sets = []
value = 0.0

[[bcs]]
name = "BC-2"
category = "DirichletBC"
type = ""
dof = ["u2"]
node_sets = ['Set-Y0']
element_sets = []
value = 0.0

[[bcs]]
name = "BC-3"
category = "DirichletBC"
type = ""
dof = ["u3"]
node_sets = ['Set-Z0']
element_sets = []
value = 0.0

[[bcs]]
name = "BC-4"
category = "DirichletBC"
type = ""
dof = ["u1"]
node_sets = ['Set-X1']
element_sets = []
value = 1.0

[solver]
type = "NonlinearSolver"
option = "NewtonRaphson"
total_time = 0.01
max_increment = 1000000
initial_dtime = 0.001
max_dtime = 1.0
min_dtime = 0.001

[[materials]]
name = "Material-1"
category = "Plastic"
type = "KinematicHardening"
data = [100000.0, 0.25, 400.0, 1000.0]

[[amplitudes]]
name = "Amp-1"
type = "TabularAmplitude"
data = [
    [0.0, 0.0],
    [1.0, 1.0]
]

[[sections]]
name = "Section-1"
category = "Solid"
type = "Volume"
option = "SmallStrain"
element_sets = ["Set-All"]
material_name = "Material-1"
data = []

[[outputs]]
type = "vtk"
field_outputs = ['S11', 'S22', 'S33', 'S12', 'S13', 'S23', 'E11', 'E22', 'E33', 'E12', 'E13', 'E23']
on_screen = false
```

### geometry file in abaqus formate:

```abaqus
*Heading
*Preprint, echo=NO, model=NO, history=NO, contact=NO
**
** PARTS
**
*Part, name=Part-1
*Node
      1,           1.,           1.,           1.
      2,           1.,           0.,           1.
      3,           1.,           1.,           0.
      4,           1.,           0.,           0.
      5,           0.,           1.,           1.
      6,           0.,           0.,           1.
      7,           0.,           1.,           0.
      8,           0.,           0.,           0.
*Element, type=C3D8R
1, 5, 6, 8, 7, 1, 2, 4, 3
*Nset, nset=Set-X0, generate
 5,  8,  1
*Elset, elset=Set-X0
 1,
*Nset, nset=Set-X1, generate
 1,  4,  1
*Elset, elset=Set-X1
 1,
*Nset, nset=Set-Y0, generate
 2,  8,  2
*Elset, elset=Set-Y0
 1,
*Nset, nset=Set-Y1, generate
 1,  7,  2
*Elset, elset=Set-Y1
 1,
*Nset, nset=Set-Z0
 3, 4, 7, 8
*Elset, elset=Set-Z0
 1,
*Nset, nset=Set-Z1
 1, 2, 5, 6
*Elset, elset=Set-Z1
 1,
*Nset, nset=Set-All, generate
 1,  8,  1
*Elset, elset=Set-All
 1,
*End Part
**  
**
** ASSEMBLY
**
*Assembly, name=Assembly
**  
*Instance, name=Part-1-1, part=Part-1
*End Instance
**  
*End Assembly
```