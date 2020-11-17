# model
model = dict(
    type='StrongBaseline',
    backbone=dict(
        type='ResNet50',
        layers=(3, 4, 6, 3),
        pretrained=False,
    ),
    classifier=dict(
        type='BaseClassifier',
        nattr=113,
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
data_root = '/opt/project/data/PA100K/'
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
    train=dict(
        type=dataset_type,
        split='train',
        ann_file=data_root + 'annotation/dataset.pkl',
        img_prefix=data_root + 'data/release_data/',
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

# log
log_level = 'INFO'
