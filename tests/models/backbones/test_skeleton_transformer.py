# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmaction.models import SkeletonTransformer
from mmaction.utils import register_all_modules


def test_skeleton_transformer_forward():
    register_all_modules()
    model = SkeletonTransformer(
        num_joints=5, num_feats=3, embed_dims=16, depth=1, num_heads=1, mlp_ratio=2)

    batch_size, num_clips, num_person, clip_len = 2, 2, 1, 4
    inputs = torch.rand(batch_size, num_clips, num_person, clip_len, 5, 3)
    outputs = model(inputs)

    assert outputs.shape == torch.Size([batch_size, 16])
    assert torch.isfinite(outputs).all()

