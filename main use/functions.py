
import torchvision
import os
import cv2
import numpy as np
import math
from skimage.morphology import skeletonize
from PIL import Image, ImageDraw
from PIL import Image, ImageDraw, ImageFont


# mask rcnn inference functions


def to_resize(img, ratio):
    w, h = img.size
    new_w = int(w/ratio)
    new_h = int(h/ratio)
    resized_img = img.resize((new_w, new_h))
    return(resized_img)




#characterization functions
def open_and_resize(root, input_masks, indx, resize_ratio):
    
    mask = cv2.imread(os.path.join(root, input_masks[indx]))  #32, 25, 45, 51
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask_r = cv2.rotate(mask, rotateCode = cv2.ROTATE_90_CLOCKWISE)
    mask_r = cv2.convertScaleAbs(mask_r, alpha= 2, beta= 0)
    scale_percent = resize_ratio # percent of original size
    width = int(mask_r.shape[1] * scale_percent )
    height = int(mask_r.shape[0] * scale_percent )
    dim = (width, height)
    # resize image
    mask_r = cv2.resize(mask_r, dim, interpolation = cv2.INTER_AREA)
    return(mask_r)

'''def sine_detection(image, w_s):
    h, w = image.shape
    search_w = int(0.5 * h)
    k_max = 0
    for line in range(10, h - 10, 1):
        sine = []
        k = 0
        top = line - search_w
        bottom = line + search_w
        if top < 0:
            top = 0
        elif bottom > h:
            bottom = h

        region = image[top:bottom, 0:w]
        # points = np.where(region>0)
        for x in range(region.shape[1]):
            for y in range(region.shape[0]):
                if region[y, x] > 0:
                    x2 = x + int(w / 2)
                    if x2 > w: x2 = x2 - w
                    y2 = line + (line - y)
                    i1 = x2 - w_s
                    i2 = x2 + w_s
                    j1 = y2 - w_s
                    j2 = y2 + w_s

                    # if i1 <0 :i1 =0  # if u change i1 or j2 change 2 
                    # if i2 >w :i2 =w
                    # if j1 <0 :j1 =0
                    # if j2 >h :j2=h
                    small_region = region[j1:j2, i1:i2]
                    if small_region.size > 0 and np.max(small_region) > 0:
                        sine.append([x, y + top])  # ???
                        k += 1
        if k > k_max:
            k_max = k
            f = sine
            loc = line

    return(f, k_max)'''
#_______________________
def get_first_depth(name, BH_name):
    if BH_name == '7220_6-1':
        text = name.split('_')
        depth_part = text[5] # last part in name
        depth = depth_part.split('_')[0]
        depth_from = depth.split('-')[0]
    elif BH_name == '7220_11-3':
        text = name.split('_')
        depth_from = text[3].split('-')[0] # depth is fourth item in the list
        depth_from = depth_from.replace(',', '.')
    return(depth_from)

def get_location(y, depth, scale, resize_ratio=1):
    location_img = y/(scale*resize_ratio) # scale in pixels per meter
    location = float(depth) + location_img
    return(location) # in m

def get_dip_angle(amp, image_width): # amp from center line (total is 2 * ampt)
    value = 2 * math.pi * amp / image_width
    angle = math.degrees(math.atan(value)) # atan alone return aangle in radian
    return(angle)

def get_dip_direction(x, sine_fit, w):
    min_sine = np.max(sine_fit)  # image y axis is inversed so we use max value to get min sine
    direction_pixels = x[np.where(sine_fit == min_sine)]
    direction_degrees = (direction_pixels * 360/w)
    return(float(direction_degrees[0]))

def crop_frac(mask, x,y):
    top = np.min(y)
    bottom = np.max(y)
    cropped = mask[top:bottom, 0:mask.shape[1]]
    return(cropped)




def read_img(img_path, resize_ratio=1):
    image = cv2.imread(img_path)
    image = cv2.rotate(image, rotateCode=cv2.ROTATE_90_CLOCKWISE)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    wi, hi, _ = image.shape
    image = cv2.resize(image,
                       (int(hi * resize_ratio), int(wi * resize_ratio)), interpolation=cv2.INTER_AREA)
    return (image)


