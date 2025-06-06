# optimizer
optimizer = dict(type='SGD', lr=0.01, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-6, by_epoch=False)


# # runtime settings
# runner = dict(type='IterBasedRunner', max_iters=80000)
# checkpoint_config = dict(by_epoch=False, interval=8000)
# evaluation = dict(interval=8000, metric='mIoU')




# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=200)
checkpoint_config = dict(by_epoch=True, interval=50, max_keep_ckpts=10)
evaluation = dict(interval=1, metric='mIoU', save_best='auto', rule='greater')


