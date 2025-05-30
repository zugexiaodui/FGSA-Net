# dataset settings
dataset_type = 'CODDataset'
data_root = '/CODData'   # 自己定义的数据集存放路径
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='COD_LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=0),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 512),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=True,
        transforms=[
            # dict(type='Resize', keep_ratio=True),
            dict(type='SETR_Resize', keep_ratio=True, crop_size=(512, 512), setr_multi_scale=True),
            dict(type='ResizeToMultiple', size_divisor=32),
            dict(type='RandomFlip', prob=1.0),       
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='TrainDataset/Imgs',
        ann_dir='TrainDataset/GT',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='TestDataset/CHAMELEON/Imgs',
        ann_dir='TestDataset/CHAMELEON/GT',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='TestDataset/CHAMELEON/Imgs',
        ann_dir='TestDataset/CHAMELEON/GT',
        # img_dir='TestDataset/CAMO/Imgs',
        # ann_dir='TestDataset/CAMO/GT',
        # img_dir='TestDataset/COD10K/Imgs',
        # ann_dir='TestDataset/COD10K/GT',
        # img_dir='TestDataset/NC4K/Imgs',
        # ann_dir='TestDataset/NC4K/GT',
        pipeline=test_pipeline))


