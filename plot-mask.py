from operator import indexOf
import cv2
import numpy as np
import os.path as osp
from math import atan2 # For computing the polar algorithm
from random import randint # For sorting
import uuid
import os
from os.path import exists
import fnmatch
import matplotlib.pyplot as plt

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

# Find the number of instances of each object in the mask
def find_instances(mask):
    unique_colors = np.unique(mask.reshape(-1, mask.shape[2]), axis=0)
    instances = []
    mask_colors = np.array([
        [219, 182, 109],    # Small Gear
        [36, 73, 146],      # Big Gear
        [146, 36, 73],      # Small Shaft
        [73, 146, 36]       # Big Shaft
    ])
    for color in unique_colors:
        if not np.array_equal(color, np.array([109, 219, 182])): # Continue of the background colour
            index_match = np.where(np.all(mask_colors == color, axis=1))
            if index_match[0].size > 0:
                print(int(index_match[0]))

                        # Find pixels with the current color
            pixels = np.all(mask == color, axis=-1)
            ys, xs = np.where(pixels)
            poly = [(x, y) for x, y in zip(xs, ys)]
            hull = graham_scan(poly)
            
            x_min, y_min, x_max, y_max = (min(xs), min(ys), max(xs), max(ys))

            instance = {
                "id": int(color[0]),
                "image_id": 1,
                "category_id": 1,
                "segmentation": [],
                "iscrowd": 0 # 0 for individual instances
            }

            # Create segmentation mask (assuming it's a polygon)
            segmentation = []
            for x,y in hull:
                segmentation.extend([float(x), float(y)])
            instance['segmentation'].append(segmentation)

            instances.append(instance)
    
    return instances

# Given the coordinates, find the perimeter of the mask
def mask_perimter(args):

    return 0

#Load the image and mask
mask = cv2.imread("./test-images-2/000000.is.png")
image = cv2.imread("./test-images-2/000000.png")

instances = find_instances(mask)

x_coords = []
y_coords = []

# Plot the image and the masks
for instance in instances:
    for polygon in instance["segmentation"]:
        polygon = np.array(polygon).reshape(-1, 2)
        x_coords.append(polygon[:, 0])
        y_coords.append(polygon[:, 1])

x_sub_coords = x_coords[0]
y_sub_coords = y_coords[0]

x_min, y_min, x_max, y_max = (min(x_sub_coords), min(y_sub_coords), max(x_sub_coords), max(y_sub_coords))

# Max and min coordinates of each bounding box
print(len(instances))

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
plt.scatter(x_sub_coords, y_sub_coords, c='red', marker='x', s=10) 
plt.title("Image with Segmentation Masks")
plt.show()

cv2.rectangle(mask, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0,0,255), 2) #BGR 
cv2.imshow("Mask with bounding box", mask)
cv2.waitKey()  
cv2.destroyAllWindows()
