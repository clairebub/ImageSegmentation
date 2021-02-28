import os
import numpy as np

import nibabel as nib
import pydicom as dicom
import matplotlib.pylab as plt
import cv2

data_path = '/data/ImageSegmentation/data/NSCLC-Radiomics/'


def generate_lung_disease_data(task='lung'):
    with open('/dataImageSegmentation/data/NSCLC-Radiomics/processed') as f:
        processed = f.read().splitlines()

    img_id = 29495

    for ct_id in os.listdir('%s/label/Thoracic_Cavities/' % data_path):

        if ct_id in processed:
            continue

        else:
            print(ct_id)

            # get mask
            mask = nib.load('%s/label/Thoracic_Cavities/%s/%s_thor_cav_primary_reviewer.nii.gz' % (data_path, ct_id, ct_id)).get_fdata()

            # convert mask to RGB
            mask_rgb = np.zeros((np.array(mask).shape[0], np.array(mask).shape[1], 3, np.array(mask).shape[2]), dtype=int)

            mask_rgb[:, :, 0, :] = np.where(mask >= 1, 1, 0)
            mask_rgb[:, :, 1, :] = np.where(mask >= 1, 1, 0)
            mask_rgb[:, :, 2, :] = np.where(mask >= 1, 1, 0)

            mask_rgb = np.rot90(mask_rgb, 1)
            mask_rgb = mask_rgb * 255

            # get dcm
            subdir_1 = os.listdir('%s/dcm/%s' % (data_path, ct_id))[0]
            subdir_2 = os.listdir('%s/dcm/%s/%s' % (data_path, ct_id, subdir_1))[0]

            for dcm_id in range(mask.shape[2]):
                if mask.shape[2] < 100:
                    dcm_file = '1-%s.dcm' % str(dcm_id + 1).zfill(2)
                else:
                    dcm_file = '1-%s.dcm' % str(dcm_id + 1).zfill(3)

                ds = dicom.dcmread('%s/dcm/%s/%s/%s/%s' % (data_path, ct_id, subdir_1, subdir_2, dcm_file))

                # save image
                os.makedirs('../data/NSCLC/%s/images' % task, exist_ok=True)
                plt.imsave('../data/NSCLC/%s/images/%d.jpg' % (task, img_id), ds.pixel_array, cmap='gray')

                # save mask
                os.makedirs('../data/NSCLC/%s/masks/0' % task, exist_ok=True)
                cv2.imwrite('../data/NSCLC/%s/masks/0/%d.png' % (task, img_id), mask_rgb[:, :, :, mask.shape[2] - dcm_id - 1])

                img_id += 1


generate_lung_disease_data()
