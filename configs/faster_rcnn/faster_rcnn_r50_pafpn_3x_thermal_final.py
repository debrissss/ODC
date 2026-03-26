_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/macvi26_thermal_od.py',
    '../_base_/schedules/schedule_3x.py',
    '../_base_/default_runtime.py'
]

custom_imports = dict(imports=['custom_modules.transforms'], allow_failed_imports=False)
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_mstrain_3x_coco/faster_rcnn_r50_fpn_mstrain_3x_coco_20210524_110822-e10bd31c.pth'

model = dict(
    # 结合消融 2：PAFPN 特征金字塔
    neck=dict(
        type='PAFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5
    ),
    roi_head=dict(
        bbox_head=dict(
            num_classes=2,
            # 结合消融 3：GIoU 损失
            reg_decoded_bbox=True, 
            loss_bbox=dict(type='GIoULoss', loss_weight=10.0)
        )
    ),
    backbone=dict(frozen_stages=-1),
    rpn_head=dict(
        anchor_generator=dict(
            # 结合消融 3：扩展 Anchor 尺寸以捕捉极小目标
            scales=[2, 4, 8],
            ratios=[0.5, 1.0, 2.0, 3.0],
            strides=[4, 8, 16, 32, 64]
        )
    )
)

# 结合消融 1：数据多尺度增强与定制热红外噪声
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
