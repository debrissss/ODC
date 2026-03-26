#!/usr/bin/env bash
# Round 4 突破瓶颈批量运行脚本
set -euo pipefail

echo "=============================="
echo "Round 4a: ATSS + COCO Checkpoint"
echo "=============================="
python tools/train.py configs/faster_rcnn/round4_atss_coco_res50.py \
    --work-dir work_dirs/round4_atss_coco_res50

echo "=============================="
echo "Round 4b: Swin-T + COCO Checkpoint"
echo "=============================="
python tools/train.py configs/faster_rcnn/round4_swin_t_coco_frcnn.py \
    --work-dir work_dirs/round4_swin_t_coco_frcnn

echo "=============================="
echo "Round 4c: Round 2 + Soft-NMS & TTA"
echo "=============================="
# 注意：作为推理端增强，方案 C 不需要训练，但由于继承了 ablation_round2_best.py，
# 建议加载 Round 2 的最佳权重进行测试
# 用法举例：python tools/test.py configs/faster_rcnn/round4_r50_inference_plus.py \
#     work_dirs/ablation_round2_best/epoch_36.pth --tta

echo "✅ Round 4 配置文件已准备就绪！"
