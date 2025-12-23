<!-- Copied/created by AI assistant: tailored, concise instructions for coding agents -->
# Copilot / AI Agent Instructions for pyfem

目的：快速让 AI 编码代理在此仓库中立即可用并产出高质量变更。

- **大局概览（Why / What）**
  - `pyfem` 是一个轻量的 Python 有限元求解器，源代码放在 `src/pyfem`（注意 `package_dir={"": "src"}`）。
  - 主要职责划分：
    - 网格与 IO：`pyfem.mesh`, `pyfem.io`（使用 `meshio` 支持多种网格格式）
    - 单元实现：`pyfem.elements` 与 `pyfem.isoelements`
    - 装配/求解流程：`pyfem.assembly`, `pyfem.fem`, `pyfem.solvers`
    - 材料本构：`pyfem.materials`

- **常用运行 / 构建 / 调试命令**
  - 从源安装并生成可执行脚本：
    - `pip install .` （仓库根目录，使用 `setup.py`）或 `python install.py`
    - 安装后可使用 `pyfem` 命令（`console_scripts` 在 `setup.py` 中注册，入口为 `pyfem.__main__:main`）。
  - 运行示例：进入示例目录并传入 toml 配置，例如：`cd examples/tutorial && pyfem -i Job-1.toml`（参见 README）。
  - 在未安装的开发环境中直接运行：`app.py` 演示了通过将 `src` 添加到 `sys.path` 启动项目的简便方式（仅用于快速调试）。
  - 文档构建：`docs/Makefile` 或 `docs/make.bat`（Windows）用于 Sphinx 文档构建。

- **环境与依赖**
  - Python >= 3.9（见 `setup.py`）。
  - 关键依赖：`numpy`, `scipy`, `meshio`, `tomli`, `tomli_w`, `h5py`, `colorlog`。

- **项目约定与惯例（重要）**
  - 源码都在 `src/pyfem`：编辑或新增包时保持此布局，测试本地变化可用 `app.py` 快速运行。
  - 版本号在 `src/pyfem/__init__.py`（更新发布版本时修改此文件）。
  - 输入算例为 TOML 文件（例：`examples/*/*.toml`），代理在修改算例或生成新算例时应遵守现有字段结构。
  - 输出格式常见为 `.pvd` / `.vtu`（ParaView 可视化），也可输出 HDF5（`h5py`）。
  - 新增单元/材料：优先在 `isoelements` 或 `elements` 下拟定模块，与现有接口（构件、形函数、积分点处理）保持风格一致。

- **代码风格与快速查找**
  - 遵循已有命名：模块与包使用小写下划线（snake_case），类使用 PascalCase。
  - 入口点：`src/pyfem/__main__.py` 包含 CLI `main()`，变更 CLI 行为请从此入手。
  - 常见工具脚本位于 `scripts/`，示例与回归算例在 `examples/`，测试脚本在 `tests/`（注意：并非全部为 pytest 风格，检查每个脚本的运行方式）。

- **可验证的快速任务示例（供代理使用）**
  - 运行并复现示例：`cd examples/tutorial && pyfem -i Job-1.toml`（检查输出 .vtu/.pvd），这能验证整体流程。
  - 增加一个简单单元测试：在 `tests/` 新增一个脚本，调用 `pyfem` 某个小模块（例如 `mesh` 或 `Job`），以确保导入路径和最小行为正确。

- **危险点与注意事项**
  - `package_dir={"": "src"}`：不要错误地移动源码出 `src/`，否则 packaging/imports 会中断。
  - 有些示例/脚本直接假设工作目录（相对路径），在自动化修改时请保留或相应调整路径。
  - 当修改 I/O 或格式（mesh、toml、h5）时，请同时更新示例以便回归验证。

如果需要，我可以把这些要点合并进 README 或生成 PR；或按你的偏好缩短/扩展某些部分。请告诉我哪些细节需要补充或删减。
