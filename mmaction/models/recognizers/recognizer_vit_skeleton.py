# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Tuple, Union

import torch
from mmengine.model import BaseModel, merge_dict

from mmaction.registry import MODELS
from mmaction.utils import (ConfigType, ForwardResults, OptConfigType,
                            OptSampleList, SampleList)


def _pool_feature(feat: Union[torch.Tensor, Tuple[torch.Tensor]]) -> torch.Tensor:
    """Pool backbone outputs to a single vector."""
    if isinstance(feat, (list, tuple)):
        feat = feat[-1]
    if feat.dim() > 2:
        feat = feat.mean(dim=tuple(range(1, feat.dim())))
    return feat


@MODELS.register_module()
class RecognizerViTSkeleton(BaseModel):
    """Two-stream recognizer that fuses ViT and skeleton transformer features."""

    def __init__(self,
                 vision_backbone: ConfigType,
                 skeleton_backbone: ConfigType,
                 cls_head: ConfigType,
                 neck: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 vision_input_key: str = 'rgb',
                 skeleton_input_key: str = 'skeleton') -> None:
        if data_preprocessor is None:
            data_preprocessor = dict(
                type='MultiModalDataPreprocessor',
                preprocessors=dict(
                    rgb=dict(type='ActionDataPreprocessor'),
                    skeleton=dict(type='ActionDataPreprocessor')))
        super().__init__(data_preprocessor=data_preprocessor)

        self.vision_input_key = vision_input_key
        self.skeleton_input_key = skeleton_input_key

        self.vision_backbone = MODELS.build(vision_backbone)
        self.skeleton_backbone = MODELS.build(skeleton_backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self.cls_head = MODELS.build(cls_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    @property
    def with_neck(self) -> bool:
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_cls_head(self) -> bool:
        return hasattr(self, 'cls_head') and self.cls_head is not None

    def extract_feat(self, inputs: Union[Tuple[torch.Tensor, torch.Tensor],
                                        Dict[str, torch.Tensor]],
                     **kwargs) -> Tuple[ForwardResults, Dict]:
        if isinstance(inputs, dict):
            vision_inputs = inputs.get(self.vision_input_key,
                                       inputs.get('rgb'))
            skeleton_inputs = inputs.get(self.skeleton_input_key,
                                         inputs.get('skeleton'))
            if vision_inputs is None or skeleton_inputs is None:
                raise KeyError('RecognizerViTSkeleton expects both vision '
                               f'and skeleton inputs, but got keys '
                               f"{list(inputs.keys())} with mapping "
                               f"{self.vision_input_key}/{self.skeleton_input_key}.")
        else:
            vision_inputs, skeleton_inputs = inputs

        vision_feat = _pool_feature(self.vision_backbone(vision_inputs))
        skeleton_feat = _pool_feature(self.skeleton_backbone(skeleton_inputs))
        feats = (vision_feat, skeleton_feat)

        loss_predict_kwargs = dict(loss_aux=dict())
        if self.with_neck:
            neck_outputs = self.neck(feats)
            if isinstance(neck_outputs, tuple) and isinstance(
                    neck_outputs[-1], dict):
                feats, loss_aux = neck_outputs
                loss_predict_kwargs['loss_aux'] = loss_aux
            else:
                feats = neck_outputs
        return feats, loss_predict_kwargs

    def loss(self,
             inputs: Union[Tuple[torch.Tensor, torch.Tensor],
                           Dict[str, torch.Tensor]],
             data_samples: SampleList,
             **kwargs) -> dict:
        feats, loss_kwargs = self.extract_feat(inputs, data_samples=data_samples)
        loss_aux = loss_kwargs.get('loss_aux', dict())
        loss_cls = self.cls_head.loss(feats, data_samples, **loss_kwargs)
        losses = merge_dict(loss_cls, loss_aux)
        return losses

    def predict(self,
                inputs: Union[Tuple[torch.Tensor, torch.Tensor],
                              Dict[str, torch.Tensor]],
                data_samples: SampleList,
                **kwargs) -> SampleList:
        feats, loss_kwargs = self.extract_feat(inputs, data_samples=data_samples)
        return self.cls_head.predict(feats, data_samples, **loss_kwargs)

    def forward(self,
                inputs: Union[Tuple[torch.Tensor, torch.Tensor],
                              Dict[str, torch.Tensor]],
                data_samples: OptSampleList = None,
                mode: str = 'tensor',
                **kwargs) -> Union[dict, SampleList, ForwardResults]:
        if mode == 'loss':
            return self.loss(inputs, data_samples, **kwargs)
        elif mode == 'predict':
            return self.predict(inputs, data_samples, **kwargs)
        elif mode == 'tensor':
            feats, _ = self.extract_feat(inputs, data_samples=data_samples,
                                         **kwargs)
            return feats
        else:
            raise ValueError(f'Invalid mode {mode}.')
