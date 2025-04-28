# Copyright (c) Shanghai AI Lab. All rights reserved.
_base_ = [
    '../_base_/models/upernet_r50.py', 
    '../_base_/datasets/COD.py',
    '../_base_/My_default_runtime.py',
    '../_base_/schedules/My_schedule.py'
]
crop_size = (512, 512)
pretrained = 'pretrained/uni-perceiver-large-L24-H1024-224size-pretrained_converted.pth'
model = dict(
    pretrained=pretrained,
    backbone=dict(
        _delete_=True,
        type='UniBaseline',
        patch_size=16,
        embed_dim=1024,
        out_indices=[5, 11, 17, 23], 
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        drop_path_rate=0.4,
        with_cp=True,  # set with_cp=True to save memory
        window_attn=[False] * 24,
        window_size=[None] * 24),
    decode_head=dict(num_classes=2, in_channels=[1024, 1024, 1024, 1024]),
    auxiliary_head=dict(num_classes=2, in_channels=1024),
    test_cfg=dict(mode='whole')
)

optimizer = dict(_delete_=True, type='AdamW', lr=6e-5, betas=(0.9, 0.999), weight_decay=0.05,
                 constructor='LayerDecayOptimizerConstructor',
                 paramwise_cfg=dict(num_layers=24, layer_decay_rate=0.8))
lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=200)
checkpoint_config = dict(by_epoch=True, interval=100, max_keep_ckpts=3)
evaluation = dict(interval=1, metric='mIoU', save_best='auto', rule='greater')
# fp16 = dict(loss_scale=dict(init_scale=512))        # 混合精度训练+动态损失放大，达到减小内存的目的。（loss_scale和loss是相乘的关系）

