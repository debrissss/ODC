#!/usr/bin/env bash
# Round 3 架构升级批量运行脚本
# 用法: bash scripts/run_round3.sh
# 按推荐顺序依次运行 A → B → C
set -euo pipefail

echo "=============================="
echo "Round 3a: ATSS + ResNet-50"
echo "=============================="
python tools/train.py configs/faster_rcnn/round3_atss_r50.py \
    --work-dir work_dirs/round3_atss_r50

echo "=============================="
echo "Round 3b: Swin-T + Faster R-CNN"
echo "=============================="
python tools/train.py configs/faster_rcnn/round3_swin_t_faster_rcnn.py \
    --work-dir work_dirs/round3_swin_t_faster_rcnn

echo "=============================="
echo "Round 3c: ATSS + Swin-T"
echo "=============================="
python tools/train.py configs/faster_rcnn/round3_atss_swin_t.py \
    --work-dir work_dirs/round3_atss_swin_t

echo "=============================="
echo "✅ Round 3 全部实验完成！"
echo "请将三个实验最终 Epoch 的验证结果发送给我进行分析。"
echo "=============================="
