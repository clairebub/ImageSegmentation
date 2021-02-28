import os
import cv2
import numpy as np
from collections import OrderedDict
import pandas as pd

from metrics import iou_score_from_array, dice_coef_from_array


def eval(disease_path, lung_path, target_path):

    disease_mask, lung_mask, target_mask = [], [], []

    for file in os.listdir(disease_path):
        tmp_disease = cv2.imread(os.path.join(disease_path, file), cv2.IMREAD_GRAYSCALE)[..., None]
        disease_mask.append(tmp_disease)

        tmp_lung = cv2.imread(os.path.join(lung_path, file), cv2.IMREAD_GRAYSCALE)[..., None]
        lung_mask.append(tmp_lung)

        tmp_target = cv2.imread(os.path.join(target_path, file), cv2.IMREAD_GRAYSCALE)[..., None]
        target_mask.append(tmp_target)

    disease_mask = np.dstack(disease_mask)
    lung_mask = np.dstack(lung_mask)
    target_mask = np.dstack(target_mask)

    disease_mask = disease_mask > 0
    lung_mask = lung_mask > 0
    target_mask = target_mask > 0

    # combine two steps: lung + disease masks
    pruned_disease_mask = disease_mask * lung_mask
    # pruned_disease_mask = disease_mask

    return OrderedDict([
        ('iou', iou_score_from_array(pruned_disease_mask, target_mask)),
        ('dice', dice_coef_from_array(pruned_disease_mask, target_mask))
    ])


disease_path = '/data/ImageSegmentation/outputs/cc_ccii_lung_disease_NestedUNet_lung_loss_woDS_testing/0'
lung_path = '/data/ImageSegmentation/data/cc_ccii/lung_disease/masks/1'
target_path = '/data/ImageSegmentation/data/cc_ccii/lung_disease/masks/0'

test_result = eval(disease_path, lung_path, target_path)

df = pd.DataFrame(test_result.items(), columns=["metric", "result"])
print(df)
