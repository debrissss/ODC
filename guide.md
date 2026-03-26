# MaCVi 2026 挑战赛 - Agent 编码与执行规范

## 1. 架构哲学与核心原则
本项目基于 MMDetection 和 MMEngine 框架，核心思想为高度模块化、拥抱框架生态、严格遵循配置继承与无头全链路可观测。必须彻底放弃“自定义流水线（Pipeline）”、“全局上下文（Context）”和“手动依赖注入”等设计模式。
- **注册表驱动（Registry-Driven）**：所有新增算法类（主干网络、数据增强、损失函数、评估指标）必须通过 MMEngine 的 Registry 机制（如 `@MODELS.register_module()`）注入框架，严禁手写业务硬编码组装或跨文件状态黑板。
- **配置即代码（Config as Code）**：超参数、训练策略和模块组装必须通过 Python 配置字典实现。严禁在业务代码中硬编码参数或手写 `argparse` 进行复杂的优先级覆盖逻辑。
- **数据流契约与生态复用**：模块间传递的核心数据必须遵循 MMDetection 标准的 `dict(inputs=..., data_samples=...)` 格式。基础计算需求优先调用 MMCV、PyTorch、Numpy 的现有高效实现，避免重复造轮子。核心算法层严禁包含 IO 操作或网络请求。

## 2. 项目目录结构与操作边界
Agent 的操作必须以官方提供的拓扑结构为基础。新增自定义模块应放置于独立的包中：
- `configs/`：模型配置文件目录。参数调优、基线修改必须在此文件夹下的 `_base_/` 与 `faster_rcnn/` 中进行。
- `tools/`：执行主入口，包含 `train.py` 和 `test.py`。除非处理版本兼容性问题（如 PyTorch 2.6 的 `_torch_load` 补丁），禁止大面积重构此目录下的核心分发逻辑。
- `data/`：**只读目录**。存放经过热极性归一化的热红外图像与 COCO 格式标注（`.json`）。严禁在此目录写入或修改原始数据。
- `scripts/`：环境构建自动化脚本（如 `install_cuda130_torch29.sh`）。
- `custom_modules/` **(运行期按需新增)**：自定义注册模块（如自定义 Transforms、Hooks 等）存放处。

*(注：原规则中有关 `docs/DEVELOPER_GUIDE.md` 文档维护的规则因脱离实际项目精简结构，已被删除。)*

## 3. 核心开发约束

### 3.1 挑战赛特定业务逻辑（Domain Specifics）
- **类别约束**：模型预测类别仅限两类：`Vessel`（船舶）与 `Navigational object`（导航目标）。
- **数据特性**：图像已经过**热极性归一化**处理（高温/发热区域像素值更高）。设计数据增强策略（Transforms）时，禁绝对其造成破坏的热力学运算（如随机反色、不合理的色彩抖动）。
- **评估导向**：核心决定指标为 COCO 标准的 **AP**，若遇平局则以 **AP50** 为准。

### 3.2 配置文件与参数合规
- **继承与覆写**：新增实验需在 `configs/faster_rcnn/` 下创建新配置，通过 `_base_ = ['../_base_/xxx.py']` 继承，只覆写需要变更的字典键值对。
- **命令行动态覆盖**：临时改变学习率或 batch size 等单次实验调整，必须通过终端的 `--cfg-options` 参数传递（例如 `--cfg-options optim_wrapper.optimizer.lr=0.01`），严禁手写 `argparse` 进行参数覆盖逻辑。

### 3.3 MMEngine Hook 探针与日志规范
- **拥抱 MMEngine Hooks**：全面废弃自定义 `print()` 或业务代码间穿插打印追踪。中间态持久化、损失记录等监控，必须通过继承 `mmengine.hooks.Hook` 实现，并在 configs 的 `custom_hooks` 中挂载。
- **生命周期拦截与无头落盘**：Hook 实例必须实现对应生命周期拦截（如 `after_train_iter`），且内部只能利用 `runner.work_dir` 设定路径进行纯文件落盘记录。严禁通过 `__init__` 构造函数强行进行实例间依赖注入探针。
- **全局日志管控**：整体层级的运行时流转日志统一由 `tools/train.py` 中的 Runner 接管。自定义组件内部必须通过 `MMLogger.get_current_instance()` 获取日志对象，且只在极端异常情况（如严重数据越界）下告警或中断报错。

### 3.4 代码风格与可读性
- 类与方法需配备标准 Google Style Docstring，标明参数类型、形状（如 `shape: (N, 4)` 边界框）及返回值。
- 行内注释仅用于解释业务动因（Why）与复杂公式（What）。避免为了注释而提取过度嵌套的逻辑，应保持早期返回（Early Return）。
- 除代码变量与专有名词外，所有注释必须使用**中文**。命名应具备物理/几何意义（如 `bboxes_img_space`，弃用模糊化的 `b_i`）。

### 3.5 核心架构运转示例 (注册与配置绑定)
Agent 在生成核心框架代码时，必须参考以下范例：

**1. 自定义数据增强 (`custom_modules/transforms.py`)**

```python
import numpy as np
from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS

@TRANSFORMS.register_module()
class ThermalNoiseInjection(BaseTransform):
    """针对热红外极性归一化图像的特定噪声注入。"""
    
    def __init__(self, noise_level: float = 0.01):
        self.noise_level = noise_level

    def transform(self, results: dict) -> dict:
        """纯函数设计，只修改并返回字典。"""
        img = results['img']
        noise = np.random.normal(0, self.noise_level, img.shape)
        results['img'] = np.clip(img + noise, 0, 255).astype(img.dtype)
        return results
```

**2. 配置驱动调用 (`configs/faster_rcnn/faster-rcnn_r50_fpn_1x_thermal.py`)**

```python
_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/thermal_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

# 引入自定义模块
custom_imports = dict(imports=['custom_modules.transforms'], allow_failed_imports=False)

# 在数据流中挂载自定义组件
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='ThermalNoiseInjection', noise_level=0.05), # 调用注册的模块
    dict(type='PackDetInputs')
]
```

**3. Hook 生命周期落盘探针 (`custom_modules/hooks.py`)**

```python
from mmengine.hooks import Hook
from mmdet.registry import HOOKS
import os

@HOOKS.register_module()
class CheckpointDebugHook(Hook):
    """纯文件驱动的中间态探针，用于无头环境的本地持久化"""
    def __init__(self, out_dir: str = 'debug_outputs'):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if self.every_n_iters(runner, 500):
            loss = outputs['loss'].item()
            with open(os.path.join(self.out_dir, "loss_log.txt"), "a") as f:
                f.write(f"Iter {runner.iter}: Loss {loss}\n")
```

## 4. 开发与运行环境约束（Local vs. Remote Linux + CUDA）
- **本地设备限制**：本地 MacBook Air (M4, 16GB) 仅用作代码编辑与文本生成。**绝对禁止生成任何针对 macOS 本地 `mps` 后端的硬件加速适配代码。**
- **远程锚定与 CUDA 强绑定**：底层依赖 MMCV (v2.1.0) 必须携带原生 CUDA 算子编译。代码在远程 Linux 主机执行，设备分配锁定 `cuda`，所有代码及脚本处理需保证与 CUDA 13.0 和 PyTorch 2.9.1 环境相融合。
- **无可视化渲染（Headless 强制阻断）**：服务器为无图形界面环境，代码逻辑中**绝对禁止**调用 `cv2.imshow()`、`plt.show()` 等阻塞式的显示 API。推断结果或测试集的画框可视化，必须通过调用平台 `tools/test.py` 时附加 `--show-dir <OUTPUT_PATH>` 参数来实现离线导出。
- **纯 POSIX 路径规范**：所有代码严格依靠 Python 的 `pathlib.Path` 或 `os.path.join` 处理路径。严禁对绝对路径硬编码或利用直接字符串拼接（如 `dir + '\\' + filename`），确保 Linux 运行的强壮性。