_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/macvi26_thermal_od.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

load_from = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_mstrain_3x_coco/faster_rcnn_r50_fpn_mstrain_3x_coco_20210524_110822-e10bd31c.pth'

# 消融实验 3：仅仅针对小目标的 Anchor 微调与 GIoU Loss 替换
model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=2,
            reg_decoded_bbox=True, 
            loss_bbox=dict(type='GIoULoss', loss_weight=10.0)
        )
    ),
    backbone=dict(frozen_stages=-1),
    rpn_head=dict(
        anchor_generator=dict(
            scales=[2, 4, 8],
            ratios=[0.5, 1.0, 2.0, 3.0],
            strides=[4, 8, 16, 32, 64]
        )
    )
)
