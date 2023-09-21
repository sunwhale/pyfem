Installation 安装
========================================

支持的操作系统包括：Windows，Linux和MacOS。

Recommend 推荐
----------------------------------------

Use the package manager `pip <https://pypi.org/project/pyfem/>`__ to
install pyfem:

使用pip命令安装：

.. prompt:: bash $

   pip install -U pyfem

If you have no root access on Linux/MacOS, please try:

如果您在Linux/MacOS上没有root访问权限，请尝试以下操作：

.. prompt:: bash $

   python -m pip install -U pyfem

Users in China can install pyfem from mirrors such as:

中国用户可以使用以下镜像：

- `Aliyun <https://developer.aliyun.com/mirror/pypi>`__

- `Tsinghua <https://mirrors.tuna.tsinghua.edu.cn/help/pypi/>`__

From Source 基于源代码
----------------------------------------

.. prompt:: bash $

   git clone https://github.com/sunwhale/pyfem.git
   cd pyfem
   pip install .

or 或者

.. prompt:: bash $

   git clone https://github.com/sunwhale/pyfem.git
   cd pyfem
   python install.py

采用第二种方法需要将可执行文件或批处理文件写入环境变量。

Quickstart 快速开始
========================================

Run in command line 在命令行运行
----------------------------------------

.. prompt:: bash $

   pyfem --help

Run the first example 执行第一个算例
----------------------------------------

当前算例文件存储目录 ``examples/tutorial`` ，该算例定义了一个二维平面应变模型，材料为塑性随动强化，载荷为y方向的循环拉伸-压缩。

.. prompt:: bash $

   cd examples/tutorial
   pyfem -i Job-1.toml

算例配置文件 Job-1.toml
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: toml

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
   is_save = true

gmsh格式的网格文件 mesh.msh
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

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

Postprocess 后处理
========================================

算例计算完成后将在配置文件所在目录下生成 ``.pvd`` 或 ``.vtu`` 文件，可以使用开源可视化软件 `paraview <https://www.paraview.org/download/>`__ 进行查看。

Preprocess 前处理
========================================

本项目暂不提供前处理模块，基于 ``meshio`` 库，可以识别\ `gmsh <https://www.gmsh.info/>`__\ 、abaqus 和 ansys等有限元软件的网格文件。
