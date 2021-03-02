import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.detection.transforms as T
import os
import PIL
from PIL import Image
import json
import numpy as np
import cv2

class fractureDataset_old(Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        self.annotations = list(sorted(os.listdir(os.path.join(root, "ann"))))  #json file here, not png 

    def __getitem__(self, idx):
        # load images
        # get reference image using annotation name 
        image_name, _ = os.path.splitext(self.annotations[idx])
        img_path = os.path.join(self.root, 'img', image_name) 
        img = Image.open(img_path).convert("RGB")
        w, h =  img.size 
        
        #load annotations
        annotations_path = os.path.join(self.root, 'ann', self.annotations[idx])
        annotations = json.load(open(annotations_path))
        
        objects = annotations['objects']  # get objects 
        #objects = objects[1:]           # remove first object (bachground)
        num_objs = len(objects)

        boxes = []
        masks = []
        for i in range(num_objs):
            pos = np.array(objects[i]['points']['exterior'])  # get points for each object (x,y)
            xmin = np.min(pos[:, 0])
            xmax = np.max(pos[:, 0])
            ymin = np.min(pos[:, 1])
            ymax = np.max(pos[:, 1])
            boxes.append([xmin, ymin, xmax, ymax])
            # creat masks
            blank = np.zeros(shape=(h, w))
            mask = cv2.fillPoly(blank, [pos], 1)
            masks.append(mask)

        masks = np.array(masks, np.int8)  
        
        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.annotations)
           
class fractureDataset(Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        self.annotations = list(sorted(os.listdir(os.path.join(root, "ann"))))  #json file here, not png 

    def __getitem__(self, idx):
        # load images
        # get reference image using annotation name 
        image_name, _ = os.path.splitext(self.annotations[idx])
        img_path = os.path.join(self.root, 'img', image_name) 
        img = Image.open(img_path).convert("RGB")
        w, h =  img.size 
        
        #load annotations
        annotations_path = os.path.join(self.root, 'ann', self.annotations[idx])
        annotations = json.load(open(annotations_path))
        
        objects = annotations['objects']  # get objects 
        #objects = objects[1:]           # remove first object (bachground)
        num_objs = len(objects)

        boxes = []
        masks = []
        for i in range(num_objs):
            pos = np.array(objects[i]['points']['exterior'])  # get points for each object (x,y)
            xmin = np.min(pos[:, 0])
            xmax = np.max(pos[:, 0])
            ymin = np.min(pos[:, 1])
            ymax = np.max(pos[:, 1])
            boxes.append([xmin, ymin, xmax, ymax])
            # creat masks
            mask = np.zeros(shape=(h, w))
            
            if objects[i]['description'] == 'augmented' :
                for point in pos:
                    mask[point[1], point[0]] = 1
            else: mask = cv2.fillPoly(mask, [pos], 1)
                
            masks.append(mask)
        masks = np.array(masks, np.int8)  
        
        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.annotations)
        
def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)