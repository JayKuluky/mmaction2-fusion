# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
import torch.nn as nn
from mmengine.model import BaseModule

from mmaction.registry import MODELS


@MODELS.register_module()
class SkeletonTransformer(BaseModule):
    """Transformer encoder for skeleton sequences.

    The module flattens the temporal and joint dimensions of a skeleton
    sequence, applies token-level positional encodings, and summarizes the
    sequence with mean pooling. The expected input shape is
    ``(B, num_clips, num_person, clip_len, num_joints, num_feats)`` where
    ``num_feats`` is typically 2 or 3.

    Args:
        num_joints (int): Number of joints in the skeleton sequence.
        num_feats (int): Number of features per joint (e.g. 2 for 2D or 3 for
            3D coordinates). Defaults to 3.
        embed_dims (int): Channel dimension of the token embeddings.
            Defaults to 256.
        depth (int): Number of transformer encoder layers. Defaults to 4.
        num_heads (int): Number of attention heads. Defaults to 8.
        mlp_ratio (float): Ratio of the feedforward hidden dimension to
            ``embed_dims``. Defaults to 4.0.
        dropout (float): Dropout probability applied inside the transformer
            encoder. Defaults to 0.1.
        max_position_embeddings (int): Maximum supported temporal length for
            positional embeddings. Defaults to 300.
        init_cfg (dict | None): Initialization config dict. Defaults to None.
    """

    def __init__(self,
                 num_joints: int,
                 num_feats: int = 3,
                 embed_dims: int = 256,
                 depth: int = 4,
                 num_heads: int = 8,
                 mlp_ratio: float = 4.,
                 dropout: float = 0.1,
                 max_position_embeddings: int = 300,
                 init_cfg: Optional[dict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.num_joints = num_joints
        self.num_feats = num_feats
        self.embed_dims = embed_dims

        self.input_proj = nn.Linear(num_feats, embed_dims)
        self.temporal_embed = nn.Embedding(max_position_embeddings, embed_dims)
        self.joint_embed = nn.Embedding(num_joints, embed_dims)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dims,
            nhead=num_heads,
            dim_feedforward=int(embed_dims * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward the skeleton transformer.

        Args:
            x (torch.Tensor): Skeleton tensor in shape
                ``(B, num_clips, num_person, clip_len, num_joints, num_feats)``.

        Returns:
            torch.Tensor: Aggregated skeleton representation with shape
            ``(B, embed_dims)``.
        """
        bsz, num_clips, num_person, clip_len, num_joints, _ = x.shape
        x = x.view(bsz * num_clips, num_person, clip_len, num_joints,
                   self.num_feats)
        x = x.reshape(-1, clip_len, num_joints, self.num_feats)

        x = self.input_proj(x)

        time_indices = torch.arange(
            clip_len, device=x.device).view(1, clip_len, 1)
        joint_indices = torch.arange(
            num_joints, device=x.device).view(1, 1, num_joints)
        pos_embed = self.temporal_embed(time_indices) + self.joint_embed(
            joint_indices)

        x = x + pos_embed
        x = x.view(x.size(0), clip_len * num_joints, self.embed_dims)
        x = self.encoder(x)
        x = self.norm(x)
        x = x.mean(dim=1)

        x = x.view(bsz, num_clips, num_person, self.embed_dims)
        x = x.mean(dim=(1, 2))
        return x
