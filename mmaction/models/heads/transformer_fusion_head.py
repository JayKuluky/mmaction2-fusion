# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch
import torch.nn as nn
from mmengine.model.weight_init import trunc_normal_

from mmaction.registry import MODELS
from .base import BaseHead


@MODELS.register_module()
class TransformerFusionHead(BaseHead):
    """Classification head for fusing ViT and skeleton transformer features.

    Args:
        num_classes (int): Number of classes to be classified.
        vision_channels (int): Channel dimension of the vision branch feature.
        skeleton_channels (int): Channel dimension of the skeleton branch
            feature.
        hidden_channels (int): Projection dimension before fusion. If not
            provided, the branch-specific channels are kept. Defaults to None.
        fusion_mode (str): Fusion strategy. Currently supports ``'concat'``
            and ``'add'``. Defaults to ``'concat'``.
        dropout (float): Dropout probability applied before classification.
            Defaults to 0.5.
        init_std (float): Initialization std for the classification layer when
            using normal init. Defaults to 0.02.
    """

    def __init__(self,
                 num_classes: int,
                 vision_channels: int,
                 skeleton_channels: int,
                 hidden_channels: int = None,
                 fusion_mode: str = 'concat',
                 dropout: float = 0.5,
                 init_std: float = 0.02,
                 **kwargs) -> None:
        fusion_channels = (hidden_channels
                           if hidden_channels is not None else vision_channels)
        super().__init__(
            num_classes=num_classes,
            in_channels=fusion_channels +
            (fusion_channels if fusion_mode == 'concat' else 0),
            **kwargs)
        self.vision_proj = nn.Linear(vision_channels, fusion_channels)
        self.skeleton_proj = nn.Linear(skeleton_channels, fusion_channels)
        self.fusion_mode = fusion_mode
        self.dropout = nn.Dropout(dropout)
        out_dim = (fusion_channels * 2 if fusion_mode == 'concat' else
                   fusion_channels)
        self.cls_head = nn.Linear(out_dim, num_classes)
        self.init_std = init_std

    def init_weights(self) -> None:
        trunc_normal_(self.cls_head.weight, std=self.init_std)
        if self.cls_head.bias is not None:
            nn.init.constant_(self.cls_head.bias, 0)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        vision_feat, skeleton_feat = x
        vision_feat = self.vision_proj(vision_feat)
        skeleton_feat = self.skeleton_proj(skeleton_feat)

        if self.fusion_mode == 'add':
            fused = vision_feat + skeleton_feat
        elif self.fusion_mode == 'concat':
            fused = torch.cat([vision_feat, skeleton_feat], dim=-1)
        else:
            raise ValueError(f'Unsupported fusion mode: {self.fusion_mode}')

        fused = self.dropout(fused)
        cls_score = self.cls_head(fused)
        return cls_score
