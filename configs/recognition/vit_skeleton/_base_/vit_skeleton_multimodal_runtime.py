_base_ = ['../../_base_/default_runtime.py']

# dataset settings
_dataset_info = dict(
    dataset_type='PoseDataset',
    ann_file='data/skeleton/ntu60_2d.pkl',
    data_root='data/nturgbd_videos/',
    left_kp=[1, 3, 5, 7, 9, 11, 13, 15],
    right_kp=[2, 4, 6, 8, 10, 12, 14, 16],
)

dataset_type = _dataset_info['dataset_type']
ann_file = _dataset_info['ann_file']
data_root = _dataset_info['data_root']
left_kp = _dataset_info['left_kp']
right_kp = _dataset_info['right_kp']

train_pipeline = [
    dict(
        type='MMUniformSampleFrames',
        clip_len=dict(RGB=8, Pose=48),
        num_clips=1),
    dict(type='MMDecode'),
    dict(type='MMCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(256, 256), keep_ratio=False),
    dict(type='RandomResizedCrop', area_range=(0.56, 1.0)),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs', collect_keys=('imgs', 'keypoint'))
]

val_pipeline = [
    dict(
        type='MMUniformSampleFrames',
        clip_len=dict(RGB=8, Pose=48),
        num_clips=1,
        test_mode=True),
    dict(type='MMDecode'),
    dict(type='MMCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(256, 256), keep_ratio=False),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs', collect_keys=('imgs', 'keypoint'))
]

test_pipeline = [
    dict(
        type='MMUniformSampleFrames',
        clip_len=dict(RGB=8, Pose=48),
        num_clips=10,
        test_mode=True),
    dict(type='MMDecode'),
    dict(type='MMCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(256, 256), keep_ratio=False),
    dict(type='ThreeCrop', crop_size=256),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs', collect_keys=('imgs', 'keypoint'))
]

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            ann_file=ann_file,
            pipeline=train_pipeline,
            split='xsub_train',
            data_prefix=dict(video=data_root))))

val_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file,
        pipeline=val_pipeline,
        split='xsub_val',
        test_mode=True,
        data_prefix=dict(video=data_root)))

# Use small batch for test to avoid OOM during validation

test_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file,
        pipeline=test_pipeline,
        split='xsub_val',
        test_mode=True,
        data_prefix=dict(video=data_root)))

val_evaluator = dict(type='AccMetric')
test_evaluator = val_evaluator

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=30, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=5e-4, weight_decay=0.05),
    paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.))

param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=True, begin=0, end=5),
    dict(type='CosineAnnealingLR', by_epoch=True, begin=5, end=30, eta_min=1e-6)
]

default_hooks = dict(checkpoint=dict(interval=5, max_keep_ckpts=2))
