_base_ = [
    './_base_/vit_skeleton_multimodal_runtime.py'
]

default_scope = 'mmaction'

model = dict(
    type='RecognizerViTSkeleton',
    data_preprocessor=dict(
        type='MultiModalDataPreprocessor',
        preprocessors=dict(
            imgs=dict(
                type='ActionDataPreprocessor',
                mean=[127.5, 127.5, 127.5],
                std=[127.5, 127.5, 127.5],
                format_shape='NCTHW'),
            keypoint=dict(type='ActionDataPreprocessor', to_float32=True))),
    vision_backbone=dict(
        type='TimeSformer',
        pretrained='https://download.openmmlab.com/mmaction/recognition/timesformer/vit_base_patch16_224.pth',
        num_frames=8,
        img_size=224,
        patch_size=16,
        embed_dims=768,
        in_channels=3,
        dropout_ratio=0.,
        transformer_layers=None,
        attention_type='divided_space_time',
        norm_cfg=dict(type='LN', eps=1e-6)),
    skeleton_backbone=dict(
        type='SkeletonTransformer',
        num_joints=17,
        num_feats=2,
        embed_dims=384,
        depth=4,
        num_heads=6,
        mlp_ratio=4.0,
        dropout=0.1,
        max_position_embeddings=200),
    cls_head=dict(
        type='TransformerFusionHead',
        num_classes=60,
        vision_channels=768,
        skeleton_channels=384,
        hidden_channels=512,
        fusion_mode='concat',
        dropout=0.5),
    vision_input_key='imgs',
    skeleton_input_key='keypoint')

# keep 8 RGB frames for TimeSformer; pose frames stay longer for smoother motion cues
train_pipeline[0]['clip_len'] = dict(RGB=8, Pose=48)
val_pipeline[0]['clip_len'] = dict(RGB=8, Pose=48)
test_pipeline[0]['clip_len'] = dict(RGB=8, Pose=48)
