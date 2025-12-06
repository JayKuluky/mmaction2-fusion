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
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                format_shape='NCTHW'),
            keypoint=dict(type='ActionDataPreprocessor', to_float32=True))),
    vision_backbone=dict(
        type='SwinTransformer3D',
        pretrained=
        'https://download.openmmlab.com/mmaction/v1.0/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k/swin_tiny_patch244_window877_kinetics400_1k.pth',
        patch_size=(2, 4, 4),
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=(8, 7, 7),
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        patch_norm=True,
        frozen_stages=-1,
        use_checkpoint=False),
    skeleton_backbone=dict(
        type='SkeletonTransformer',
        num_joints=17,
        num_feats=2,
        embed_dims=384,
        depth=4,
        num_heads=6,
        mlp_ratio=4.0,
        dropout=0.1,
        max_position_embeddings=400),
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

# use smaller clips to match swin tiny default sampling
train_pipeline[0]['clip_len'] = dict(RGB=32, Pose=48)
val_pipeline[0]['clip_len'] = dict(RGB=32, Pose=48)
test_pipeline[0]['clip_len'] = dict(RGB=32, Pose=48)
