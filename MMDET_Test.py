import argparse
import cv2
import mmcv 
import torch 

from mmdet.apis import inference_detector, init_detector
from mmdet.utils import register_all_modules
from mmdet.registry import VISUALIZERS

# Path to the config and checkpoint file
config_file = r'./Config/custom-config.py'
checkpoint_file = r'./Weights/epoch_300.pth'

# Build model from config file and check point file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# Initialize visualizer
visualizer = VISUALIZERS.build(model.cfg.visualizer)
# the dataset_meta is loaded from the checkpoint and # then pass to the model in init_detector
visualizer.dataset_meta = model.dataset_meta

# Initialize camera
device_id = 0
cam = cv2.VideoCapture(device_id)
pred_thresh = 0.5

#Run  main loop
while True:
    ret, img = cam.read()
    result = inference_detector(model, img)

    img = mmcv.imconvert(img, 'bgr', 'rgb')
    visualizer.add_datasample(
        name='result',
        image=img,
        data_sample=result,
        draw_gt=False,
        pred_score_thr=pred_thresh,
        show=False)
    
    img = visualizer.get_image()
    img=mmcv.imconvert(img, 'bgr', 'rgb')
    cv2.imshow('result', img)

    key = cv2.waitKey(1)
    if key == 27 or key == ord('q'):
        break