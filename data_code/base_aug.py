#!/usr/bin/env python
# coding: utf-8



import matplotlib.pyplot as plt
import glob
import json
import PIL
from PIL import ImageDraw
import os
import numpy as np
from PIL import Image, ExifTags, ImageEnhance
import random
from resizeimage import resizeimage
from statistics import mean
import cv2 


# # Base images augmentation

def open_img (file):
    image =  Image.open(file)
    
    for orientation in ExifTags.TAGS.keys() : 
        if ExifTags.TAGS[orientation]=='Orientation' : break 

    exif=dict(image._getexif().items())

    if   exif[orientation] == 3 : 
        image=image.rotate(180, expand=True)
    elif exif[orientation] == 6 : 
        image=image.rotate(270, expand=True)
    elif exif[orientation] == 8 : 
        image=image.rotate(90, expand=True)
        
    return(image)

def flip(img):
    f = random.randint(1, 3)
    if f == 1: img = img # no flip
    elif f == 2: img = img.transpose(PIL.Image.FLIP_TOP_BOTTOM) #h_flip
    else: img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT) #v_flip
    return(img)


def rand_crop_sine(img):
    W, H = img.size
    ratio_w = (random.randint(25, 50)/100)    #0.25 to 0.5 
    ratio_h = (random.randint(25, 50)/100)  # different ratios for w and h to get different scales
    w, h = int(W * ratio_w), int(H * ratio_h)
    x1 = random.randint(0, (W-w))
    y1 = random.randint(0, (H-h))
    img = img.crop((x1, y1, x1+w, y1+h))
    return(img)

def resize(img):
    big_axis = max(img.size)
    if big_axis > 600:
        f = big_axis/600
        w, h = int(img.width / f), int(img.height / f)
        img = resizeimage.resize_cover(img, [w, h])
    else: pass
    return(img)

def contrast(img):
    factor = random.randint(7, 20)/10
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(factor)
    return(img)
    

def brightness(img):
    factor = random.randint(5, 15)/10
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(factor)
    return(img)


def aug_base_sine(img):
    img = flip(img)
    img = rand_crop_sine(img)
    img = contrast(img)
    img = brightness(img)
    img = resize(img)
    return (img)

def rand_crop_real(img):
    ratio = (random.randint(6, 9)/10)
    w, h = int(img.width * ratio), int(img.height * ratio)
    img = resizeimage.resize_crop(img, [w, h])
    return(img)

def aug_base_real(img):
    img = flip(img)
    img = rand_crop_real(img)
    return (img)