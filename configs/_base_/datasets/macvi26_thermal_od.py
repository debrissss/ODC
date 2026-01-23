# COCO-style dataset configuration

# Dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/'

classes=('Vessel', 'Navigational object')

# Backend settings for file reading  
backend_args = None

# Data pipeline for training
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomResize',
        scale=(800, 600),
        ratio_range=(0.8, 1.0),
        keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

# Data pipeline for validation
val_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(800, 600), keep_ratio=True, backend='pillow'),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')
    )
]

# Data pipeline for testing
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(800, 600), keep_ratio=True, backend='pillow'),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')
    )
]

# Dataloader settings
train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='instances_train.json',
        data_prefix=dict(img='train/'),
        metainfo=dict(classes=classes),
        filter_cfg=dict(filter_empty_gt=True, min_size=0),
        pipeline=train_pipeline,
        backend_args=backend_args
    )
)

val_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='instances_val.json',
        data_prefix=dict(img='val/'),
        metainfo=dict(classes=classes),
        test_mode=True,
        pipeline=val_pipeline,
        backend_args=backend_args
    )
)

test_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='instances_test_empty.json',
        data_prefix=dict(img='test/'),
        metainfo=dict(classes=classes),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args
    )
)

# Evaluator for COCO-style metrics
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'instances_val.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args
)

test_evaluator = dict(
    type='CocoMetric',
    ann_file='data/instances_test_empty.json',
    format_only=True,
    outfile_prefix='results_test',
    backend_args=backend_args
)
