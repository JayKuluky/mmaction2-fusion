#!/usr/bin/env python3
# Copyright (c) OpenMMLab. All rights reserved.
"""Run fused ViT + skeleton transformer inference on a video and keypoints."""
import argparse
import os
from typing import List, Sequence

import mmcv
import mmengine
import numpy as np
import torch
from mmengine.registry import init_default_scope

from mmaction.apis import init_recognizer
from mmaction.structures import ActionDataSample
from mmaction.utils import register_all_modules


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMAction2 ViT + skeleton transformer fusion demo')
    parser.add_argument('video', help='Path to an RGB video file')
    parser.add_argument('skeleton', help='Path to skeleton keypoints (.npy/.pkl)')
    parser.add_argument(
        '--config',
        default='demo/demo_configs/vit_skeleton_fusion_demo.py',
        help='Model config file')
    parser.add_argument(
        '--checkpoint',
        default=None,
        help='Optional checkpoint to load for the fusion model')
    parser.add_argument(
        '--device',
        default='cuda:0',
        help='Device used for inference (e.g., "cuda:0" or "cpu")')
    parser.add_argument(
        '--clip-len',
        type=int,
        default=16,
        help='Number of frames sampled from the video')
    parser.add_argument(
        '--target-size',
        type=int,
        default=224,
        help='Shorter side will be resized to this value')
    parser.add_argument(
        '--label-map',
        default=None,
        help='Optional label map file to convert prediction indices to text')
    parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Number of top classes to show')
    return parser.parse_args()


def _load_video_frames(path: str, clip_len: int, target_size: int) -> torch.Tensor:
    """Load frames uniformly from a video and return (C, T, H, W) tensor."""
    if not os.path.exists(path):
        raise FileNotFoundError(f'Video path does not exist: {path}')

    reader = mmcv.VideoReader(path)
    if not reader:
        raise RuntimeError(f'Cannot open video: {path}')
    total = len(reader)
    indices = np.linspace(0, total - 1, num=clip_len, dtype=np.int64)
    frames: List[np.ndarray] = []
    for idx in indices:
        frame = reader[idx]
        if frame is None:
            raise RuntimeError(f'Failed to read frame {idx} from {path}')
        frame = mmcv.imresize(frame, (target_size, target_size))
        frame = mmcv.imconvert(frame, 'bgr', 'rgb')
        frames.append(frame)

    # (T, H, W, C) -> (C, T, H, W)
    frame_array = np.stack(frames).transpose(3, 0, 1, 2)
    return torch.from_numpy(frame_array)


def _load_skeleton(path: str, clip_len: int) -> torch.Tensor:
    """Load skeleton keypoints and reshape to (1, num_person, T, V, C)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f'Skeleton path does not exist: {path}')

    if path.endswith('.npy'):
        data = np.load(path, allow_pickle=True)
    elif path.endswith('.pkl'):
        data = mmengine.load(path)
    else:
        raise ValueError('Skeleton file must be .npy or .pkl')

    if isinstance(data, list):
        data = np.array(data, dtype=object)
    if isinstance(data, dict):
        if 'keypoint' in data:
            data = data['keypoint']
        else:
            raise KeyError('Expected "keypoint" field in skeleton dict')

    arr = np.array(data)
    if arr.ndim == 3:
        # (T, V, C) -> (1, T, V, C)
        arr = arr[None, ...]
    elif arr.ndim == 4:
        # (num_person, T, V, C)
        pass
    else:
        raise ValueError('Skeleton array must have shape (T, V, C) or '
                         '(num_person, T, V, C)')

    num_person, total_frames, num_joints, num_feats = arr.shape
    if total_frames < clip_len:
        pad = np.repeat(arr[:, -1:], clip_len - total_frames, axis=1)
        arr = np.concatenate([arr, pad], axis=1)
    elif total_frames > clip_len:
        indices = np.linspace(0, total_frames - 1, num=clip_len, dtype=np.int64)
        arr = arr[:, indices]

    # Add num_clips dimension and reorder to (num_clips, num_person, T, V, C)
    arr = arr[None, ...]
    return torch.from_numpy(arr)


def _maybe_load_labels(path: str) -> Sequence[str]:
    if path is None:
        return []
    if not os.path.exists(path):
        raise FileNotFoundError(f'Label map not found: {path}')
    with open(path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def main():
    args = parse_args()
    register_all_modules()

    cfg = mmengine.Config.fromfile(args.config)
    init_default_scope(cfg.get('default_scope', 'mmaction'))
    model = init_recognizer(cfg, args.checkpoint, args.device)

    rgb = _load_video_frames(args.video, args.clip_len, args.target_size)
    skeleton = _load_skeleton(args.skeleton, args.clip_len)

    data_batch = {
        'inputs': {
            'rgb': [rgb],
            'skeleton': [skeleton]
        },
        'data_samples': [ActionDataSample()]
    }

    with torch.no_grad():
        preds = model.test_step(data_batch)[0]

    scores = preds.pred_score.cpu().numpy()
    label_names = _maybe_load_labels(args.label_map)

    top_k = min(args.top_k, scores.shape[-1])
    top_indices = scores.argsort()[::-1][:top_k]
    print('\nTop predictions:')
    for rank, idx in enumerate(top_indices, 1):
        label = label_names[idx] if label_names else f'class_{idx}'
        print(f'{rank}. {label}: {scores[idx]:.4f}')


if __name__ == '__main__':
    main()
