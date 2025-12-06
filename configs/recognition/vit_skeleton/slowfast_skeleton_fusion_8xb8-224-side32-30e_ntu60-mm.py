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
        type='ResNet3dSlowFast',
        pretrained=None,
        resample_rate=4,
        speed_ratio=4,
        channel_ratio=8,
        slow_pathway=dict(
            type='resnet3d',
            depth=50,
            pretrained=None,
            lateral=True,
            conv1_kernel=(1, 7, 7),
            dilations=(1, 1, 1, 1),
            conv1_stride_t=1,
            pool1_stride_t=1,
            inflate=(0, 0, 1, 1),
            norm_eval=False),
        fast_pathway=dict(
            type='resnet3d',
            depth=50,
            pretrained=None,
            lateral=False,
            base_channels=8,
            conv1_kernel=(5, 7, 7),
            conv1_stride_t=1,
            pool1_stride_t=1,
            norm_eval=False)),
    skeleton_backbone=dict(
        type='SkeletonTransformer',
        num_joints=17,
        num_feats=2,
        embed_dims=320,
        depth=4,
        num_heads=5,
        mlp_ratio=4.0,
        dropout=0.1,
        max_position_embeddings=400),
    cls_head=dict(
        type='TransformerFusionHead',
        num_classes=60,
        vision_channels=2304,
        skeleton_channels=320,
        hidden_channels=512,
        fusion_mode='concat',
        dropout=0.5),
    vision_input_key='imgs',
    skeleton_input_key='keypoint')

# align sampling to SlowFast default of 32 frames (4x slow, 32x fast)
train_pipeline[0]['clip_len'] = dict(RGB=32, Pose=48)
val_pipeline[0]['clip_len'] = dict(RGB=32, Pose=48)
test_pipeline[0]['clip_len'] = dict(RGB=32, Pose=48)
