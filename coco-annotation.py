from operator import indexOf
import cv2
import numpy as np
import os.path as osp
from mmengine.fileio import dump
import mmcv
from math import atan2 # For computing the polar algorithm
from random import randint # For sorting
import uuid
import os
from os.path import exists
import fnmatch
from tqdm import tqdm
import time

###############################################
## Graham Scan
###############################################

def polar_angle(p0, p1=None):
    if p1==None: p1=anchor
    y_span=p0[1]-p1[1]
    x_span=p0[0]-p1[0]
    return atan2(y_span, x_span)

def distance(p0, p1=None):
    if p1==None: p1=anchor
    y_span=p0[1]-p1[1]
    x_span=p0[0]-p1[0]
    return y_span**2 + x_span**2

# If +det - 3 points represent cw turn
# If -det - 3 points represent ccw turn
# if 0, 3 points are co linear
def det(p1, p2, p3): #This function helps to determine
    return (p2[0]-p1[0])*(p3[1]-p1[1]) \
        -(p2[1]-p1[1])*(p3[0]-p1[0])

def quicksort(a):
    if len(a)<=1: return a
    smaller,equal,larger=[],[],[]
    piv_ang=polar_angle(a[randint(0, len(a)-1)])
    for pt in a:
        pt_ang=polar_angle(pt)
        if pt_ang<piv_ang:      smaller.append(pt)
        elif pt_ang==piv_ang:   equal.append(pt)
        else:                   larger.append(pt)
    return      quicksort(smaller) \
                +sorted(equal, key=distance) \
                +quicksort(larger)

def graham_scan(points):
    global anchor
    min_idx=None
    for i, (x,y) in enumerate(points):
        if min_idx==None or y<points[min_idx][1]:
            min_idx=i
        if y==points[min_idx][1] and x<points[min_idx][0]:
            min_idx=i

    anchor=points[min_idx]
    sorted_pts=quicksort(points)
    del sorted_pts[sorted_pts.index(anchor)]

    hull=[anchor,sorted_pts[0]]
    for s in sorted_pts[1:]:
        while det(hull[-2], hull[-1], s)<=0:
            del hull[-1] # backtrack
            if len(hull)<2: break
        hull.append(s)
    return hull

###############################################
## Gather image info
###############################################

def image_data(image_prefix, filename): # THIS FUNCTION TAKES IN IMAGE AND OUTPUTS DATA ASSOCIATED WITH THAT IMAGE
    img_path = osp.join(image_prefix, filename)
    height, width = mmcv.imread(img_path).shape[:2]
    idx = int(filename.split('.')[0])
    filename = filename.replace(".is", '')

    images.append(
        dict(id=idx, file_name=filename, height=height, width=width))
    
    return images, idx, img_path

###############################################
## Find number of instances in mask
###############################################

def find_instances(img_path, idx):
    mask = cv2.imread(img_path)
    unique_colors = np.unique(mask.reshape(-1, mask.shape[2]), axis=0)
    mask_colors = np.array([
        [168, 170, 170],    # Small Gear
        [254, 255, 127],      # Big Gear
    ])
    for color in unique_colors:
         if np.any(np.all(mask_colors == color, axis=1)): #THIS IS IN BGR VALUE

            # Find the category associated with the color
            index_match = np.where(np.all(mask_colors == color, axis=1))

            # Find pixels with the current color
            pixels = np.all(mask == color, axis=-1)
            ys, xs = np.where(pixels)
            poly = [(x, y) for x, y in zip(xs, ys)]
            hull = graham_scan(poly)
            
            x_min, y_min, x_max, y_max = (min(xs), min(ys), max(xs), max(ys))

            # Store data annotations in a dictionary
            data_anno = dict(
                image_id= idx, # Image ID
                id=uuid.uuid1().int>>64, # Unique ID for the annotation
                category_id = int(index_match[0]), # Category ID
                bbox=[x_min, y_min, x_max - x_min, y_max - y_min],
                area = (x_max - x_min) * (y_max - y_min),
                segmentation=[[coord for point in hull for coord in point]],
                iscrowd=0)

            # Append annotations
            annotations.append(data_anno)
    
    return annotations

###############################################
## Create annotation format
###############################################

def coco_annotation_generator(images, annotations, out_file):
    coco_format_json = dict(
        images = images,
        annotations = annotations,
        categories=[{
            'id': 0,
            'name':'Small Gear'
        },
        {
            'id': 1,
            'name': 'Big Gear'
        }])
    
    dump(coco_format_json, out_file)
    
if __name__ == '__main__':
    start_time = time.time()  # Record the start time
    annotations = []
    images = []
    image_prefix =r'data' # Directory where images are
    out_file = 'annotation_coco.json' # Name of the output file
    count = 0 
    tracker = 0

    # Count Number of Image Segmentation Files
    for root, dirs, files in os.walk(image_prefix):
        for filename in files:
            if filename.endswith('.is.png'):
                count += 1
    print(f"{count} Segmentation Files were Found!")

    #Initalize progress bar
    progress_bar = progress_bar = tqdm(files, desc="Processing Files", unit="file")

    # Loop through each image seg file
    for filename in files:
        if filename.endswith('.is.png'):
            images, idx, img_path = image_data(image_prefix, filename)
            annotations = find_instances(img_path, idx)

        progress_bar.update(1)
    
    # Close progress bar once done
    progress_bar.close()

    # Write annotation file
    print("Writing to file...")
    coco_annotation_generator(images, annotations, out_file)
    end_time = time.time()  # Record the end time
    execution_time = end_time - start_time  # Calculate the execution time
    # Print the execution time in seconds
    print(f"Execution time: {execution_time:.2f} seconds")