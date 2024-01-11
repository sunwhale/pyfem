# pyfem

pyfem是一个完全基于python语言实现的极简有限元求解器。依赖的第三方库包括numpy、scipy和meshio等，主要用于有限元方法的学习、有限元算法验证和快速建立材料本构模型的程序原型。


[![Codacy Badge](https://app.codacy.com/project/badge/Grade/ab5bca55d85d45d4aa4336ccae058316)](https://app.codacy.com/gh/sunwhale/pyfem/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)

## Installation 安装

支持的操作系统包括：Windows，Linux和MacOS。

### Recommend 推荐

Use the package manager [pip](https://pypi.org/project/pyfem/) to install pyfem:

使用pip命令安装:

```bash
pip install pyfem
```

If you have no root access on Linux/MacOS, please try
```bash
python -m pip install pyfem
```

Users in China can install pyfem from mirrors such as:

中国用户可以使用以下镜像:
- [Aliyun](https://developer.aliyun.com/mirror/pypi)
- [Tsinghua](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/)

### From Source 基于源代码

```bash
git clone https://github.com/sunwhale/pyfem.git
cd pyfem
pip install .
```

or 或者

```bash
git clone https://github.com/sunwhale/pyfem.git
cd pyfem
python install.py
```
采用第二种方法需要将可执行文件或批处理文件写入环境变量。

## Quickstart 快速开始

### Run in command line 在命令行运行:

```bash
pyfem --help
```

### Run the first example 执行第一个算例:

当前算例文件存储目录 examples/tutorial，该算例定义了一个二维平面应变模型，材料为塑性随动强化，载荷为y方向的循环拉伸-压缩。

```bash
cd examples/tutorial
pyfem -i Job-1.toml
```

## 后处理

算例计算完成后将在配置文件所在目录下生成 .pvd 或 .vtu文件，可以使用开源可视化软件 [paraview](https://www.paraview.org/download/) 进行查看。

## 前处理

本项目暂不提供前处理模块，基于 meshio 库，可以识别[gmsh](https://www.gmsh.info/)、abaqus 和 ansys等有限元软件的网格文件。

## Documents 帮助文档
[https://pyfem-doc.readthedocs.io/](https://pyfem-doc.readthedocs.io/)

## Development

### ToDo list

- [x] 增加Neumann边界条件，支持concentrated force，distributed和pressure定义方式
- [ ] 增加hdf5计算结果输出格式
- [ ] 完善帮助文档
- [ ] 完善输入文件的校检
- [x] 增加测试模块
- [x] 增加日志模块
- [x] 增加后台运行模式
- [ ] 处理平面应力状态的面外应力平衡
- [x] 增加粘弹性力学本构模型
- [x] 增加晶体塑性力学本构模型
- [x] 增加温度场求解单元
- [x] 增加温度场-位移场耦合求解单元
- [x] 增加相场-位移场耦合求解单元
- [ ] 增加内聚区单元
- [ ] 增加动力学求解器
- [ ] 建立前处理界面

### Bug list

- [ ] 采用abaqus网格文件时，如果存在node不属于任何element则在计算时会导致全局刚度矩阵奇异。
