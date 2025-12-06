# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmaction.models import TransformerFusionHead
from mmaction.structures import ActionDataSample
from mmaction.utils import register_all_modules


def test_transformer_fusion_head_concat_and_loss():
    register_all_modules()
    head = TransformerFusionHead(
        num_classes=3,
        vision_channels=4,
        skeleton_channels=6,
        hidden_channels=8,
        fusion_mode='concat',
        dropout=0.)
    head.init_weights()

    feats = (torch.rand(2, 4), torch.rand(2, 6))
    cls_scores = head(feats)
    assert cls_scores.shape == torch.Size([2, 3])

    data_samples = [ActionDataSample().set_gt_label(1) for _ in range(2)]
    losses = head.loss(feats, data_samples)
    assert 'loss_cls' in losses


def test_transformer_fusion_head_add():
    register_all_modules()
    head = TransformerFusionHead(
        num_classes=2,
        vision_channels=5,
        skeleton_channels=7,
        hidden_channels=9,
        fusion_mode='add',
        dropout=0.)
    head.init_weights()

    feats = (torch.rand(3, 5), torch.rand(3, 7))
    cls_scores = head(feats)
    assert cls_scores.shape == torch.Size([3, 2])

