import nibabel as nib
import os
import cv2
import numpy as np


def generate_disease_data(task='disease'):
    # read img and mask

    img_id = 1
    for file in os.listdir('/data/covid19_internal/covid_ct_segmentation/data/mosmed/COVID19_1110/masks'):
        img_slices = nib.load('/data/covid19_internal/covid_ct_segmentation/data/mosmed/COVID19_1110/studies/CT-1/%s' % file.replace('_mask', '')).get_fdata()
        mask_slices = nib.load('/data/covid19_internal/covid_ct_segmentation/data/mosmed/COVID19_1110/masks/%s' % file).get_fdata()

        for i in range(img_slices.shape[2]):

            mask = mask_slices[:, :, i]
            mask_ids = np.unique(mask)

            if mask_ids.shape[0] == 2:

                # generate and save CT image
                img = img_slices[:, :, i]
                img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8)
                img_rgb = np.zeros((np.array(img).shape[0], np.array(img).shape[1], 3), dtype=int)
                img_rgb[:, :, 0] = img
                img_rgb[:, :, 1] = img
                img_rgb[:, :, 2] = img
                img_rgb = np.rot90(img_rgb, 1)

                if not os.path.exists('../data/mosmed/%s/images' % task):
                    os.makedirs('../data/mosmed/%s/images' % task)
                cv2.imwrite('../data/mosmed/%s/images/%d.jpg' % (task, img_id), img_rgb)

                # generate and save CT mask
                mask_rgb = np.zeros((np.array(mask).shape[0], np.array(mask).shape[1], 3), dtype=int)
                mask_rgb[:, :, 0] = np.where(mask == 1, 1, 0)
                mask_rgb[:, :, 1] = np.where(mask == 1, 1, 0)
                mask_rgb[:, :, 2] = np.where(mask == 1, 1, 0)
                mask_rgb = np.rot90(mask_rgb, 1)
                mask_rgb = mask_rgb * 255

                if not os.path.exists('../data/mosmed/%s/masks/0' % task):
                    os.makedirs('../data/mosmed/%s/masks/0' % task)
                cv2.imwrite('../data/mosmed/%s/masks/0/%d.png' % (task, img_id), mask_rgb)

                img_id += 1


generate_disease_data()
