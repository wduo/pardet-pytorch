# model
model = dict(
    type='StrongBaseline',
    backbone=dict(
        type='ResNet50',
        layers=(3, 4, 6, 3),
        pretrained=True,
    ),
    classifier=dict(
        type='BaseClassifier',
        nattr=26,
    ),
    loss=dict(
        type='CEL_Sigmoid',
        sample_weight=None,
        size_average=True
    ),
)
train_cfg = dict(
    train_cfg='',
)
test_cfg = dict(
    test_cfg='',
)

# data
dataset_type = 'PA100K'
data_root = '/pardet-pytorch/data/PA100K/'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(type='Resize', size=(256, 192)),
    dict(type='Pad', padding=10),
    dict(type='RandomCrop', size=(256, 192)),
    dict(type='RandomHorizontalFlip', p=0.5),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
]
test_pipeline = [
    dict(type='Resize', size=(256, 192), keep_ratio=True),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
]
data = dict(
    batchsize=2,
    workers=2,
    shuffle=True,
    train=dict(
        type=dataset_type,
        split='train',
        ann_file=data_root + 'annotation/dataset.pkl',
        img_prefix=data_root + 'data/release_data/release_data/',
        pipeline=train_pipeline
    ),
    val=dict(
        type=dataset_type,
        split='val',
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline
    )
)
evaluation = dict(interval=1, metric='bbox')

# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
total_epochs = 12

# logs
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# ckpts
checkpoint_config = dict(interval=1)

# misc
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

