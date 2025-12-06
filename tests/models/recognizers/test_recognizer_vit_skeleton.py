# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import MagicMock

import torch
import torch.nn as nn
from mmengine.model import BaseModule

from mmaction.registry import MODELS
from mmaction.structures import ActionDataSample
from mmaction.utils import register_all_modules


@MODELS.register_module(force=True)
class DummyVisionBackbone(BaseModule):

    def __init__(self, out_channels: int = 8) -> None:
        super().__init__()
        self.proj = nn.Linear(3, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect input shape (B, C, H, W)
        pooled = x.mean(dim=(2, 3))
        return self.proj(pooled)


@MODELS.register_module(force=True)
class DummySkeletonBackbone(BaseModule):

    def __init__(self, out_channels: int = 8) -> None:
        super().__init__()
        self.proj = nn.Linear(3, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect input shape (B, num_clips, num_person, T, V, C)
        pooled = x.mean(dim=(1, 2, 3, 4))
        return self.proj(pooled)


def test_recognizer_vit_skeleton_train_and_test():
    register_all_modules()
    model_cfg = dict(
        type='RecognizerViTSkeleton',
        data_preprocessor=dict(
            type='MultiModalDataPreprocessor',
            preprocessors=dict(
                rgb=dict(type='ActionDataPreprocessor', to_float32=True),
                skeleton=dict(type='ActionDataPreprocessor', to_float32=True))),
        vision_backbone=dict(type='DummyVisionBackbone', out_channels=6),
        skeleton_backbone=dict(type='DummySkeletonBackbone', out_channels=6),
        cls_head=dict(
            type='TransformerFusionHead',
            num_classes=4,
            vision_channels=6,
            skeleton_channels=6,
            fusion_mode='add',
            dropout=0.))

    recognizer = MODELS.build(model_cfg)
    data_batch = {
        'inputs': {
            'rgb': [torch.rand(3, 4, 4)],
            'skeleton': [torch.rand(1, 1, 4, 5, 3)]
        },
        'data_samples': [ActionDataSample().set_gt_label(2)]
    }

    optim_wrapper = MagicMock()
    loss_vars = recognizer.train_step(data_batch, optim_wrapper)
    assert 'loss' in loss_vars
    optim_wrapper.update_params.assert_called_once()

    with torch.no_grad():
        predictions = recognizer.test_step(data_batch)
    score = predictions[0].pred_score
    assert score.shape == torch.Size([4])
    assert torch.min(score) >= 0
    assert torch.max(score) <= 1

