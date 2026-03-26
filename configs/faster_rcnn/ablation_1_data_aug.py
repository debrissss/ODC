_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/macvi26_thermal_od.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

custom_imports = dict(imports=['custom_modules.transforms'], allow_failed_imports=False)
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_mstrain_3x_coco/faster_rcnn_r50_fpn_mstrain_3x_coco_20210524_110822-e10bd31c.pth'

# 还原基础模型的 Head 配置
model = dict(
    roi_head=dict(bbox_head=dict(num_classes=2)),
    backbone=dict(frozen_stages=-1),
    rpn_head=dict(
        anchor_generator=dict(
            scales=[4, 8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]
        )
    )
)

# 消融实验 1：仅仅替换数据增强与多尺度训练
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomResize',
        scale=[(1333, 600), (1333, 960)],
        keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='ThermalNoiseInjection', noise_level=0.03),
    dict(type='PackDetInputs')
]

train_dataloader = dict(
    dataset=dict(
        pipeline=train_pipeline
    )
)
