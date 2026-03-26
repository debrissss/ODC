_base_ = [
    './ablation_round2_best.py'
]

# Round 4 方案 C：Round 2 最优版 + 推理端增强
# 在不重新训练的前提下，通过 Soft-NMS 处理密集重叠目标，提升定位精度

model = dict(
    test_cfg=dict(
        rcnn=dict(
            score_thr=0.05,
            # 将标准 NMS 替换为 Soft-NMS
            nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05),
            max_per_img=100)
    )
)

# TTA (测试时增强) 配置建议在 tools/test.py 时通过 --tta 参数开启
# 或在此处显式定义（MMDet 3.x 推荐做法）
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
