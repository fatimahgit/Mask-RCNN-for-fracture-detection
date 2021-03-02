import numpy as np
import torch
import torch.nn as nn


def cal_IOU(mask1, mask2):
    intersection= np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou_score = np.sum(intersection) / np.sum(union)
    return(iou_score)


def get_loss (mask1, mask2):
    loss_ = nn.BCELoss(reduction='mean')
    output = loss_(mask1, mask2)
    return(output)

def evaluate_masks(test_images, model, device, save = False):
    # to save the results
    results ={}
    results['images'] = []
    # to plot the average values
    k = 0
    total_loss = 0
    total_iou = 0

    for image, target in test_images:
        image_results = {}
        image_results['image_id'] = target['image_id']
        image_results['masks'] = []
        image_results['iou'] = []
        image_results['loss'] = []
        image_results['score'] = []
        image_results['gt_masks'] = []

        image = image.to(device)
        predictions = model([image])
        predicted_masks = predictions[0]['masks'].to('cpu').detach().numpy()
        predicted_scores = predictions[0]['scores'].to('cpu').detach().numpy()
        gt_masks = target['masks'].detach().numpy()  # tensoe, cpu

        for mask, score in zip(predicted_masks, predicted_scores):
            mask = mask[0]
            max_iou = 0
            for gt_mask in gt_masks:
                if gt_mask.shape == mask.shape:
                    iou = cal_IOU(mask, gt_mask)
                    if iou >= max_iou:
                        max_iou = iou
                        matched_gt = gt_mask
            loss_mask = get_loss(torch.from_numpy(mask), torch.from_numpy(matched_gt.astype(np.float32)))
            k += 1
            total_loss += loss_mask
            total_iou += max_iou

            if save == True:
                image_results['masks'].append(mask)
                image_results['iou'].append(max_iou)
                image_results['loss'].append(loss.detach().numpy())
                image_results['score'].append(score)
                image_results['gt_masks'].append(matched_gt)

        results['images'].append(image_results)
    avg_loss = total_loss/k
    avg_iou = total_iou/k
    return (results, avg_loss.detach().numpy().item(), avg_iou.item())

def evaluate_masks_two_stages(test_images, model, device, save = False):
    # to save the results
    results ={}
    results['images'] = []
    # to plot the average values
    k = 0
    total_loss = 0
    total_iou = 0

    for image, target in test_images:
        image_results = {}
        image_results['image_id'] = target['image_id']
        image_results['masks'] = []
        image_results['iou'] = []
        image_results['loss'] = []
        image_results['score'] = []
        image_results['gt_masks'] = []

        image = image.to(device)
        predictions = model([image])
        predicted_masks = predictions[0]['masks'].to('cpu').detach().numpy()
        predicted_scores = predictions[0]['scores'].to('cpu').detach().numpy()
        gt_masks = target['masks'].detach().numpy()  # tensoe, cpu

        for mask, score in zip(predicted_masks, predicted_scores):
            mask = mask[0]
            max_iou = 0
            for gt_mask in gt_masks:
                if gt_mask.shape == mask.shape:
                    iou = cal_IOU(mask, gt_mask)
                    if iou >= max_iou:
                        max_iou = iou
                        matched_gt = gt_mask
            loss_mask = get_loss(torch.from_numpy(mask), torch.from_numpy(matched_gt.astype(np.float32)))
            k += 1
            total_loss += loss_mask
            total_iou += max_iou

            if save == True:
                image_results['masks'].append(mask)
                image_results['iou'].append(max_iou)
                image_results['loss'].append(loss.detach().numpy())
                image_results['score'].append(score)
                image_results['gt_masks'].append(matched_gt)

        results['images'].append(image_results)
    avg_loss = total_loss/k
    avg_iou = total_iou/k
    return (results, avg_loss.detach().numpy().item(), avg_iou.item())