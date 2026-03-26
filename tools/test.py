#!/usr/bin/env python
"""
Testing/Inference script for object detection models using MMDetection.

This script evaluates trained models on test datasets or runs inference
on individual images.

Usage:
    python tools/test.py <config_file> <checkpoint> [options]

Example:
    python tools/test.py configs/faster_rcnn/faster_rcnn_r50_fpn_1x.py work_dirs/faster_rcnn/epoch_12.pth
"""

import argparse
import os
import os.path as osp
import sys

# 将项目根目录加入 sys.path，确保 custom_modules 等自定义包可被正确导入
sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), '..')))

from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

import torch

# Monkeypatch torch.load BEFORE MMEngine tries to load checkpoints because in PyTorch 2.6 the default value of the `weights_only` argument in `torch.load` changed from `False` to `True`.
_orig_torch_load = torch.load
def _torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _torch_load

from mmengine.logging.history_buffer import HistoryBuffer
torch.serialization.add_safe_globals([HistoryBuffer])


def parse_args():
    parser = argparse.ArgumentParser(
        description='Test (and optionally evaluate) a detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--out',
        type=str,
        help='dump predictions to a pickle file for offline evaluation')
    parser.add_argument(
        '--show', action='store_true', help='show prediction results')
    parser.add_argument(
        '--show-dir',
        help='directory where painted images will be saved. '
        'If specified, it will be automatically saved '
        'to the work_dir/timestamp/show_dir')
    parser.add_argument(
        '--wait-time', type=float, default=2, help='the interval of show (s)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--tta', action='store_true', help='Test time augmentation')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    # Load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher

    # Override config options
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # Set work directory
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    if args.tta:
        if 'tta_model' not in cfg:
            raise RuntimeError('Cannot find "tta_model" in config.')
        if 'tta_pipeline' not in cfg:
            raise RuntimeError('Cannot find "tta_pipeline" in config.')

        cfg.model = dict(**cfg.tta_model, module=cfg.model)
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline

    # Load checkpoint
    cfg.load_from = args.checkpoint

    # Build the runner from config
    if 'runner_type' not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)

    # Run testing
    runner.test()


if __name__ == '__main__':
    main()
