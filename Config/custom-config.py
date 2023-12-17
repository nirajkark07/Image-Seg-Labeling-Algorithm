import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# The new config inherits a base config to highlight the necessary modification
_base_ = './rtmdet-ins_l_8xb32-300e_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    backbone=dict(deepen_factor=1.33, widen_factor=1.25),
    neck=dict(
        in_channels=[320, 640, 1280], out_channels=320, num_csp_blocks=4),
    bbox_head=dict(in_channels=320, 
                   feat_channels=320,
                   num_classes=5))

# Modify dataset related settings
data_root = 'data/'
metainfo = {
    'classes': ('Small Gear', 'Big Gear', 'Small Shaft', 'Big Shaft', 'Ball Bearing'),
    'palette': [
        (220, 20, 60),
        (150, 10, 20),
        (59, 10, 11),
        (42, 55, 69),
        (69, 69, 69),
    ]
}
train_dataloader = dict(
    batch_size=5,
    num_workers=2,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='Annotations/train.json',
        data_prefix=dict(img='Train/')))
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='Annotations/val.json',
        data_prefix=dict(img='Val/')))
test_dataloader = val_dataloader

# Modify metric related settings
val_evaluator = dict(ann_file=data_root + 'Annotations/val.json')
test_evaluator = val_evaluator

# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = './checkpoints/rtmdet-ins_x_8xb16-300e_coco_20221124_111313-33d4595b.pth'