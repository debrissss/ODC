_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/macvi26_thermal_od.py',
    '../_base_/schedules/schedule_3x.py',
    '../_base_/default_runtime.py'
]

# Round 3 方案 B：Swin-T 骨干 + Faster R-CNN
# 核心改进：用 Swin Transformer 替换 ResNet-50，获得全局上下文感知能力
# 同时保留 Round 2 验证有效的 GIoU Loss

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'

model = dict(
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(in_channels=[96, 192, 384, 768]),
    roi_head=dict(
        bbox_head=dict(
            num_classes=2,
            reg_decoded_bbox=True,
            loss_bbox=dict(type='GIoULoss', loss_weight=10.0))),
    rpn_head=dict(
        anchor_generator=dict(
            scales=[4, 8],
            ratios=[0.5, 1.0, 2.0, 3.0],
            strides=[4, 8, 16, 32, 64])))

# Swin Transformer 需要 AdamW 优化器，并对位置编码和 LayerNorm 禁用权重衰减
optim_wrapper = dict(
    type='OptimWrapper',
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)}),
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=0.0001,
        betas=(0.9, 0.999),
        weight_decay=0.05))
