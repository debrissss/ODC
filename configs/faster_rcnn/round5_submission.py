_base_ = [
    './round5_swin_t_inference_plus.py'
]

# Round 5 最终提交配置 (针对 Test 集)
# 使用 Swin-T + TTA + Soft-NMS 生成竞赛提交文件

test_dataloader = dict(
    dataset=dict(
        ann_file='instances_test_empty.json',
        data_prefix=dict(img='test/'),
        pipeline=_base_.tta_pipeline))

test_evaluator = dict(
    type='CocoMetric',
    ann_file='data/instances_test_empty.json',
    metric='bbox',
    format_only=True, # 必须为 True，因为 Test 集没有真值标签
    outfile_prefix='work_dirs/round5_final_submission/results')
