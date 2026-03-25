# 📁 MaCVi 2026 - 热红外目标检测挑战赛 项目说明

本项目是 CVPR 2026 举办的第4届海事计算机视觉研讨会（MaCVi 2026）中“热红外目标检测挑战赛 (Thermal Object Detection Challenge)”的官方入门代码库（Starter Repo）。该挑战赛旨在提升无人水面艇（USVs）在夜间和低能见度环境下的自主导航与辅助导航能力。

## 1. 🌟 项目概述
- **核心任务**：对热红外（IR）图像中的目标进行检测，输出轴对齐的边界框（Bounding boxes）和目标类别标签。
- **目标类别**：为缓解类别不平衡问题，类别被简化为两类：
  - **船舶 (Vessel)**：包括机动船和帆船等。
  - **导航目标 (Navigational object)**：包括航标、障碍物等。
- **数据源与预处理**：数据集源自德国、英吉利海峡及荷兰海域采集的碰撞避免数据集，本挑战赛仅采用其中的热红外（IR）图像数据。此外，所有图像已进行**热极性归一化**处理，统一将温度较高（发热）区域映射为更高的像素值。
- **评价协议**：采用 COCO 目标检测评价协议，核心决定指标为 **AP**（平均精度），若遇平局则以 **AP50** 作为最终排名依据。
- **奖励设置**：第一名获胜团队将获得一块由 catskill 赞助的 NVIDIA RTX 5080 GPU。

---

## 2. 📂 项目结构与文件说明

结合 `README.md` 与官方 HTML 网页的介绍以及项目目录结构，项目的各个文件和目录的作用如下：

### 📄 根目录文件
- **`README.md`**：项目的官方操作说明指南，包含环境搭建步骤、数据集目录组织结构、Faster R-CNN R50-FPN 基线模型的训练和测试运行说明，以及生成提交格式的指导。
- **`MaCVi @ CVPR 2026 _ Maritime Computer Vision Workshop.html`**：挑战赛的官方网页存档。包含详细的挑战赛设立背景、任务详情、数据集类的映射逻辑配置、提交文件格式要求、评估算法说明及挑战赛的服务条款。
- **`requirements.txt`**：Python 运行环境的纯依赖列表配置文件，列出了基础的 Python 包（例如 `ninja`, `psutil`, `pybind11` 等辅助编译工具以及基础数据处理扩展）。
- **`LICENSE`**：项目的开源许可协议。
- **`.gitignore`**：Git 版本控制软件的忽略规则配置，用于排除数据集、权重文件、日志等无需推送到远端服务器的文件。

### 📁 核心目录结构
- **`assets/`**：存放项目描述和示例文档所引用的静态资源（如 `thermal_challenge_example_train.png`）。
- **`configs/`**：MMDetection 模型框架的配置文件目录。
  - `_base_/`：存放检测器的基础模板配置，包括数据集配置、模型骨架配置、学习率调节策略、数据增强等基础构成。
  - `faster_rcnn/`：具体实验设置下的模型配置文件，本项目提供的官方检测基线配置（Faster R-CNN）存放在该文件夹中，主配置将继承组合 `_base_` 里的设定。
- **`data/`**：存放测试和训练数据集的根目录。官方提供的 Zip 包应解压至此目录下，并包含以下特定层级：
  - `train/` / `val/` / `test/` （存放 `.png` 热红外图像）。
  - `instances_train.json` / `instances_val.json` （COCO 标注格式的文件，验证和训练集专用）。
- **`scripts/`**：存放供开发者自动化配置与管理相关操作的 Bash 运行脚本。
- **`tools/`**：包含进行机器学习模型训练、推理与验证阶段的入口 Python 源码和控制工具脚本。

---

## 3. 💻 核心代码文件及方法说明

### 🛠️ `scripts/install_cuda130_torch29.sh`
- **作用**：提供在 Linux 发行版下的自动化装配脚本。该脚本专门面向配置包含 PyTorch 2.9.1 及 CUDA 13.0 的环境编译。
- **核心逻辑**：
  - 更新基础 pip，并安装 `requirements.txt` 内的环境依赖。
  - 手动安装 `mmengine` (v0.10.7)。
  - 从源码仓库下载并编译携带原生 CUDA 算子的 `MMCV` (v2.1.0) 工具箱，以避免系统底层版本不一致导致的环境崩溃。
  - 安装 `mmdet` (v3.3.0) 库，并在最后环节对 PyTorch 和 MMCV 基础模块进行 `import` 测试以确保安装成功和 CUDA 高效联动。

### 🚀 `tools/train.py`
- **作用**：模型的训练主入口脚本，启动基于 MMDetection 配置字典参数的深度学习目标检测模型训练流程。
- **核心方法 / 函数**：
  - `parse_args()`：命令行参数解析函数。参数包含：
    - `config`：传入的配置脚本文件路径。
    - `--work-dir`：输出日志、权重模型的存储位置。
    - `--amp`：是否触发自动混合精度训练（AMP）。
    - `--auto-scale-lr`：启用后根据 batch-size 自动等比例缩放配置表中的学习率。
    - `--resume`：用于从指定路径加载中断的保存点，以恢复中断的训练过程。
    - 其它进阶支持分布式参数(`--launcher`, `--local_rank`)及自定义参数覆盖覆盖传递(`--cfg-options`)。
  - `main()`：入口主控逻辑方法。
    1. 调用 `setup_cache_size_limit_of_dynamo()` 来调整编译缓存，优化网络训练速度。
    2. 基于 CLI 初始化 `Config`，动态覆盖配置参数，并推断生成默认的工作输出保存路径（`work_dir`）。
    3. 解析 `--amp` 与 `--auto-scale-lr` 标志旗，注入混合精度参数和自适应学习率策略到网络初始化中。
    4. 执行 `register_all_modules(init_default_scope=True)`，完整注册 MMDetection 底层作用域，让框架正常读取并实例化算法注册表。
    5. 调用 `Runner.from_cfg(cfg)` 创建负责训练流程中调度与日志分发的核心引擎套件 (`runner`)。最后执行 `runner.train()` 进入循环训练。

### 🎯 `tools/test.py`
- **作用**：全流程测试与推断验证入口。可加载已训练完毕的模型权重 (checkpoint) 在验证集/挑战赛测试集上生成评测结果或导出 COCO 规定格式预测字典（供打榜上传竞赛服务器使用）。
- **全局补丁机制 (Monkeypatch)**：由于 PyTorch 2.6 安全机制的更新，此文件顶部重写并拦截了标准 `torch.load` 函数逻辑为自定义 `_torch_load`，强制传递 `weights_only=False` 参数以读取之前带有运行时复杂对象的 MMEngine checkpoint 检查点。同时将 `HistoryBuffer` 类型加入安全白名单对象。
- **核心方法 / 函数**：
  - `parse_args()`：命令行参数解析函数。支持项除 `config`、`checkpoint` 等必需件外，还包括：
    - `--out`：生成推断后的离线反序列化预测文件 (.pkl)。
    - `--show` 与 `--show-dir`：直接开启终端窗口实时预览检测框或将可视化的预测结果画框后导出保存到目录下。
  - `main()`：推断与数据生成挂载方法。
    1. 动态加载环境网络配置文件信息，并指定注入 `cfg.load_from` 使用模型传入的评测模型预训练检查点载入位置。
    2. 使用 `Runner.from_cfg(cfg)` 构建 MMEngine 推断引擎模块，调用 `runner.test()` 从评估器（如预设的 `CocoMetric`）中提取并触发推断流程。在此挑战赛预设中，该指令会自动在工作目录输出如 `results_test.bbox.json` 的提交物。
