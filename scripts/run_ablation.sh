#!/usr/bin/env bash
# 消融实验批量运行脚本
# 用法: bash scripts/run_ablation.sh
# 每个实验的训练日志和权重将保存到各自的 work_dirs 子目录下
set -euo pipefail

echo "=============================="
echo "消融实验 1/4: 数据增强与多尺度"
echo "=============================="
python tools/train.py configs/faster_rcnn/ablation_1_data_aug.py \
    --work-dir work_dirs/ablation_1_data_aug

echo "=============================="
echo "消融实验 2/4: PAFPN 架构改进"
echo "=============================="
python tools/train.py configs/faster_rcnn/ablation_2_pafpn.py \
    --work-dir work_dirs/ablation_2_pafpn

echo "=============================="
echo "消融实验 3/4: 小目标 Anchor + GIoU"
echo "=============================="
python tools/train.py configs/faster_rcnn/ablation_3_small_object.py \
    --work-dir work_dirs/ablation_3_small_object

echo "=============================="
echo "消融实验 4/4: 3x Schedule (36 Epochs)"
echo "=============================="
python tools/train.py configs/faster_rcnn/ablation_4_schedule_3x.py \
    --work-dir work_dirs/ablation_4_schedule_3x

echo "=============================="
echo "✅ 全部消融实验完成！"
echo "请将以下四个日志文件中最终 Epoch 的验证结果发送给我进行分析："
echo "  work_dirs/ablation_1_data_aug/"
echo "  work_dirs/ablation_2_pafpn/"
echo "  work_dirs/ablation_3_small_object/"
echo "  work_dirs/ablation_4_schedule_3x/"
echo "=============================="
