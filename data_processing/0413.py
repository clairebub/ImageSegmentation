import os
import cv2
import numpy as np

import nibabel as nib

data_path = '/data/covid19_internal/covid_ct_segmentation/data/0413'


def generate_disease_data(task='0413'):
    # read img and mask

    img = nib.load('%s/tr_im.nii.gz' % data_path).get_fdata()
    mask = nib.load('%s/tr_mask.nii.gz' % data_path).get_fdata()

    for i in range(img.shape[2]):

        # generate image
        slice = cv2.normalize(img[:, :, i], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8)

        slice_rgb = np.zeros((np.array(slice).shape[0], np.array(slice).shape[1], 3), dtype=int)
        slice_rgb[:, :, 0] = slice
        slice_rgb[:, :, 1] = slice
        slice_rgb[:, :, 2] = slice
        slice_rgb = np.rot90(slice_rgb, 3)

        if not os.path.exists('../data/cc_ccii/%s/images' % task):
            os.makedirs('../data/cc_ccii/%s/images' % task)
        cv2.imwrite('../data/cc_ccii/%s/images/%d.jpg' % (task, i), slice_rgb)

        # generate mask
        slice_mask = mask[:, :, i]

        slice_mask[slice_mask < 2] = 0
        slice_mask[slice_mask >= 2] = 1

        slice_mask = cv2.normalize(slice_mask, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8)
        slice_mask_rgb = np.zeros((np.array(slice_mask).shape[0], np.array(slice_mask).shape[1], 3), dtype=int)
        slice_mask_rgb[:, :, 0] = slice_mask
        slice_mask_rgb[:, :, 1] = slice_mask
        slice_mask_rgb[:, :, 2] = slice_mask
        slice_mask_rgb = np.rot90(slice_mask_rgb, 3)

        if not os.path.exists('../data/cc_ccii/%s/masks/0' % task):
            os.makedirs('../data/cc_ccii/%s/masks/0' % task)
        cv2.imwrite('../data/cc_ccii/%s/masks/0/%d.png' % (task, i), slice_mask_rgb)


generate_disease_data()
