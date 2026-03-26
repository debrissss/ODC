_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/macvi26_thermal_od.py',
    '../_base_/schedules/schedule_3x.py',
    '../_base_/default_runtime.py'
]

# 严格遵循注册表与配置即代码规范引入自定义模块
custom_imports = dict(imports=['custom_modules.transforms'], allow_failed_imports=False)

# 加载预训练权重
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_mstrain_3x_coco/faster_rcnn_r50_fpn_mstrain_3x_coco_20210524_110822-e10bd31c.pth'

# 1. 网络架构与小目标微调: 修改 Anchor 尺寸与添加 GIoU Loss
model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=2,
            reg_decoded_bbox=True, # 必须设定为 True 才能使用 GIoULoss 回归预测框的真实坐标
            loss_bbox=dict(type='GIoULoss', loss_weight=10.0)
        )
    ),
    backbone=dict(
        frozen_stages=-1
    ),
    rpn_head=dict(
        anchor_generator=dict(
            # 加入更小的尺度 scale 2 以捕获远处热源小目标，同时配置 ratios 拟合不同视角的长条形船只
            scales=[2, 4, 8],
            ratios=[0.5, 1.0, 2.0, 3.0],
            strides=[4, 8, 16, 32, 64]
        )
    )
)

# 2. 数据处理与多尺度增强
backend_args = None
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    # 多尺度缩放 (Multi-scale Training) 提升图像采样率改善小目标可见度
    dict(
        type='RandomResize',
        scale=[(1333, 600), (1333, 960)],
        keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    # 挂载定制化的红外噪声注入，不对图像结构进行反转破坏
    dict(type='ThermalNoiseInjection', noise_level=0.03),
    dict(type='PackDetInputs')
]

# 在继承了 dataset 基础配置的情况下使用 Python 字典部分覆写 train_pipeline
train_dataloader = dict(
    dataset=dict(
        pipeline=train_pipeline
    )
)
