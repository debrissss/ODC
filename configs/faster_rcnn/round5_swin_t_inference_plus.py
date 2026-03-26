_base_ = [
    './round4_swin_t_coco_frcnn.py'
]

# Round 5 方案 A：Swin-T 巅峰版 (Swin-T + TTA + Soft-NMS)
# 结合了 Swin-T 强大的大目标检测能力与 TTA/Soft-NMS 的小目标补强

model = dict(
    test_cfg=dict(
        rcnn=dict(
            score_thr=0.05,
            # 将标准 NMS 替换为 Soft-NMS，处理重叠船舶
            nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05),
            max_per_img=100)
    )
)

# TTA (测试时增强) 显式定义
tta_model = dict(
    type='DetTTAModel',
    tta_cfg=dict(
        nms=dict(type='soft_nms', iou_threshold=0.5),
        max_per_img=100))

tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale=(1333, 800), keep_ratio=True),
                dict(type='Resize', scale=(1000, 600), keep_ratio=True),
                dict(type='Resize', scale=(1666, 1000), keep_ratio=True),
            ],
            [
                dict(type='RandomFlip', prob=1.),
                dict(type='RandomFlip', prob=0.),
            ],
            [
                dict(
                    type='PackDetInputs',
                    meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                               'scale_factor', 'flip', 'flip_direction'))
            ]
        ])
]

# === 关键：切换到 Val 集进行指标验证 ===
test_dataloader = dict(
    dataset=dict(
        ann_file='instances_val.json',
        data_prefix=dict(img='val/'),
        pipeline=tta_pipeline))

test_evaluator = dict(
    type='CocoMetric',
    ann_file='data/instances_val.json',
    metric='bbox',
    format_only=False)
