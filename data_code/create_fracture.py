import PIL
from PIL import ImageDraw
import os
import numpy as np
from PIL import Image, ExifTags
import random
import math
from statistics import mean, median
import cv2 
import json

# Masks constraction
def get_masks(ann):
    '''
     to genearte binary seperated masks from an annotation file,
     generated mask sizeequal to the size of the bbox (to include that object only)
    '''
    annotations = json.load(open(ann))
    image_size = annotations['size']
    h = image_size['height']
    w = image_size["width"]
    objects = annotations['objects']  # get objects
    #objects = objects[1:]           # remove first object (bachground)
    num_objs = len(objects)

    masks = []
    for i in range(num_objs):
        pos = np.array(objects[i]['points']['exterior'])  # get points for each object (x,y)
        xmin = np.min(pos[:, 0])
        xmax = np.max(pos[:, 0])
        ymin = np.min(pos[:, 1])
        ymax = np.max(pos[:, 1])
        blank = np.zeros(shape=(h, w))
        mask = cv2.fillPoly(blank, [pos], 1)
        # add a margine for cropping, for the width only
        r = 100 #pixels
        mask = mask[ymin:ymax, xmin-r:xmax+r]
        masks.append(mask)

    #masks = np.array(masks, np.int8)
    return(masks)

#____________ augmentation of  real fractures_______

def open_mask(path):
    # open and convert to 1 channel mask
    mask = cv2.imread(path)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    # randon flipping
    f = random.randint(0, 2)  # 0 for flip vertically, 1 for flip horizontall, 2 for no change
    if f == 0 or f == 1:
        mask = cv2.flip(mask, f)
    else:
        mask = mask
    return (mask)


def decrease_thickness(mask):
    k = random.randint(3, 5)
    kernel = np.ones((k, k), np.uint8)
    new_thick = cv2.erode(mask, kernel, iterations=1)
    return (new_thick)


def calculate_thickness(mask):
    if np.max(mask) > 1: mask = mask / 255
    thickness = median([sum(mask[y, :]) for y in range(mask.shape[0])])
    return (thickness)


def change_thickness(mask):
    thickness = calculate_thickness(mask)
    operation = random.choice(['no_change', 'thinner'])
    if (operation == 'thinner' and thickness > 7):
        new_thick = decrease_thickness(mask)
    else:
        new_thick = mask
    return (new_thick)


def decrease_w(mask):
    ratio = (random.randrange(6, 9)) / 10  # range 0.4 to 0.8
    width = int(mask.shape[1] * ratio)
    height = mask.shape[0]  # keep original height
    new_width = cv2.resize(mask, (width, height), interpolation=cv2.INTER_AREA)
    return (new_width)


def resize_width(mask):
    operation = random.choice(['no_change', 'decrease_w'])
    if operation == 'decrease_w':
        new_width = decrease_w(mask)
    else:
        new_width = mask
    return (new_width)


#___________________ sine shaped frac
def sine(x, y, amp, b, phase):
    return y+ amp * math.sin(b * (x -phase))

def creat_sine(base_image, loc = None, amp_h = None):
    '''
    loc: fracture centerline location as a percentage of the image height
    amp_h: amplitude as a pecentage of image hight, maximum value of 0.5
    '''
    w, h = base_image.shape[1], base_image.shape[0]
    phase = random.randint(0, w)
    if loc is None:
        loc = random.randint(25, 75)/100 # as a fraction of h
    shift = abs(loc-0.5) # the shift from the center line of the image
    if amp_h is None:
        amp_h = random.randint(10, 40)/100 #amp as a percentage of h
    amp = abs(int((amp_h-shift)*h))
    y = int(loc*h)
    b = 2* math.pi/w
    points_x = [i for i in range (0, w)]
    points_y = [sine(x, y, amp, b, phase) for x in points_x]
    return(points_x, points_y)

def get_segmets(points_x, points_y):
    total_x = []
    total_y = []
    seg_len = int(len(points_x)/random.randint(1,4))

    for k in range (0, len(points_x), seg_len):
        seg_x = points_x[k:k+seg_len]
        seg_y = points_y[k:k+seg_len]
        seg_aperture = random.randint(1,8)
        l = len(seg_x)
        for i in range (0, int(0.5 * l)+1):
            add=int(0.1 *i)
            if add > seg_aperture: add = seg_aperture
            #if i% 25 ==0: r = random.randint(-2,2) # add some randon noise (len of 10 pixels)
            for j in range(add):
                seg_x.append(seg_x[i])
                seg_y.append(seg_y[i]+j)
                seg_x.append(seg_x[i])
                seg_y.append(seg_y[i]-j)
                
        for i in range (l, int(0.5 * l), -1):
            add= int(0.1 * (l- i))
            if add > seg_aperture: add = seg_aperture
            #if i% 25 ==0: r = random.randint(-2,2)
            for j in range(add):
                seg_x.append(seg_x[i])
                seg_y.append(seg_y[i]+j) #+r
                seg_x.append(seg_x[i])
                seg_y.append(seg_y[i]-j)
        
        total_x.extend(seg_x)
        total_y.extend(seg_y)

    return(total_x, total_y)



def create_initial_mask(img, points_x, points_y):
    w, h = img.shape[1], img.shape[0] 
    mask = np.zeros((h,w))
    for y,x in zip(points_y, points_x):
        mask[int(y),x] = 1
    return(mask)

