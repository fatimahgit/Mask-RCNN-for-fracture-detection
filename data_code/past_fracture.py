import PIL
from PIL import ImageDraw
import os
import numpy as np
from PIL import Image, ExifTags
import random
from resizeimage import resizeimage
from statistics import mean
import cv2
import copy

def match_img_mask (img, mask, before = None):
    # x axis
    diff_x = img.shape[1] - mask.shape[1]
    if before == None : before = random.randrange(0, diff_x, 100)
    after = img.shape[1]- mask.shape[1]- before
    mask = np.pad(mask, ((0,0), (before, after)), 'edge')

    # macth y axis
    diff_y = abs(mask.shape[0]- img.shape[0])
    top = int(diff_y/2)
    if img.shape[0] < mask.shape[0]:
        mask = mask[top:top+img.shape[0], 0:mask.shape[1]]
    else:
        img = img[top:top+mask.shape[0], 0:img.shape[1]]
    return(img, mask)


def clip_image(image, value):
    '''
    to remove very bright pixel values and
     clip to a lower (darker) value
    '''
    image2 = np.copy(image)
    for i in range(image2.shape[2]):
        image2[:,:,i] = np.clip(image2[:,:,i] , 0, value)
    return(image2)


def calculate_thickness(mask):
    # calculate the aperture of the fracture
    if np.max(mask) > 1 : mask = mask/255
    thickness = mean([sum(mask[y, :]) for y in range (mask.shape[0])])
    return(thickness)

def creat_edges(mask, edges):
    '''
    to create some gradient in the fracture before pasting on image
    (outer edges have less intensity)
    '''
    thickness = calculate_thickness(mask)
    k = int(thickness/4)
    kernel = np.ones((k,k),np.uint8)
    edge_intensity = 1-(2*edges)/10 #1 - edges/10 #outer edge intensity
    mask_before = mask
    blank = np.zeros((mask.shape[0], mask.shape[1]), np.float)
    for i in range(edges +1):
        mask_after = cv2.erode(mask_before,kernel)
        edge = mask_before - mask_after
        edge = edge * edge_intensity
        blank = blank + edge
        #edges.append(edge)
        #points = np.where(edge != 0)
        #edges_points.append(points)
        mask_before = mask_after
        edge_intensity += 0.2
    blank = blank + mask_after
    return(blank)


def generate_image(base, mask, sigma_x, intensity, k_d, iter_, edges, clipping_value):
    # the main function to generater a fracture image
    thickness = calculate_thickness(mask)
    print('thick',thickness)
    #mask = cv2.blur(mask,(int(thickness/3),int(thickness/3))) #blur edges better than guassian
    mask = creat_edges(mask, edges)
    k = int(thickness/k_d) # or bigger
    if k% 2 != 1: k = k+1
    if k < 3 : k =3
    for i in range (iter_):
        mask =  cv2.GaussianBlur(mask,(k,k) , sigmaX = sigma_x, sigmaY = sigma_x)  
    #add edge here:
    
    mask = mask * intensity
    base = base.astype(np.float)
    base_clipped = clip_image(base, clipping_value)
    base_blured = cv2.blur(base_clipped, (5,5)) #clip_image(base, clipping_value)
    
    mask = mask * 255
    mask = mask.astype(np.float)
    points = np.where(mask > 1)
    for i in range (base.shape[2]):
        for p in range (len(points[0])):
            y = points[0][p]
            x = points[1][p]
            base [y, x, i] = np.clip(base_blured [y, x, i] - mask [y, x] , 0.0, 255.0)
    return(base.astype(np.uint8), thickness)

# create annotation file for the generated image, it modifies an existed json file (for simplisity)
def create_ann_json(base_file, mask):
    new_ann = base_file
    new_ann['size']['height'] = mask.shape[0]
    new_ann['size']['width'] = mask.shape[1]
    templet = new_ann['objects'][0]
    new_objects = []
    # get objects from masks
    np_points = np.where(mask != 0)
    x = np_points[1]
    y = np_points[0]
    points = [[int(x[i]), int(y[i])] for i in range (len(x))]
    add_obj = templet
    add_obj['points']['exterior']= points
    add_obj['description'] = 'augmented'
    new_objects.append(add_obj)
    new_ann['objects'] = new_objects
    return(new_ann)

# similar function in case of keeping the first fracture (when creating intersection)
def create_ann_json_intersection(base_file, mask):
    new_ann = base_file
    new_ann['size']['height'] = mask.shape[0]  
    new_ann['size']['width'] = mask.shape[1]
    existing_obj_list = list.copy(new_ann['objects'])
    np_points = np.where(mask != 0)
    x = np_points[1]
    y = np_points[0]
    points = [[int(x[i]), int(y[i])] for i in range(len(x))]
    add_obj = copy.deepcopy(existing_obj_list[0])
    add_obj['points']['exterior'] = []
    add_obj['points']['exterior'] = points
    add_obj['description'] = 'augmented'
    new_ann['objects'].append(add_obj)  
    return (new_ann)