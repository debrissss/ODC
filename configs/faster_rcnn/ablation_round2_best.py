_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/macvi26_thermal_od.py',
    '../_base_/schedules/schedule_3x.py',
    '../_base_/default_runtime.py'
]

load_from = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_mstrain_3x_coco/faster_rcnn_r50_fpn_mstrain_3x_coco_20210524_110822-e10bd31c.pth'

# === Round 2 最优组合 ===
# 策略：仅保留消融实验中被验证为正增益的组件
# ✅ 3x Schedule (ΔmAP +0.030) — 通过 _base_ 继承
# ✅ GIoU Loss (ΔmAP +0.023 来自消融3) — 保留
# ✅ 扩展 ratios [0.5, 1.0, 2.0, 3.0] — 保留
# ❌ scale=2 — 去掉（消融3中 mAP_l 下降0.034的元凶）
# ❌ PAFPN — 回退为 FPN（消融2中 mAP_l 下降0.056）
# ❌ 大幅多尺度/ThermalNoise — 回退（消融1中 mAP 暴跌0.111）

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
            scales=[4, 8],
            ratios=[0.5, 1.0, 2.0, 3.0],
            strides=[4, 8, 16, 32, 64]
        )
    )
)
