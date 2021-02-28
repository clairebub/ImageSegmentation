import numpy as np
import torch
import torch.nn.functional as F


def iou_score(output, target):

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5

    return iou_score_from_array(output_, target_)


def dice_coef(output, target):

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5

    return dice_coef_from_array(output_, target_)


def iou_score_from_array(prediction, target):
    smooth = 1e-5

    intersection = (prediction * target).sum()
    # union = (prediction | target).sum()
    union = prediction.sum() + target.sum() - intersection

    return (intersection + smooth) / (union + smooth)


def dice_coef_from_array(prediction, target):
    smooth = 1e-5

    intersection = (prediction * target).sum()

    return (2. * intersection + smooth) / \
        (prediction.sum() + target.sum() + smooth)

