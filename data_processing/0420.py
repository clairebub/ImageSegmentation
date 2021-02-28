import nibabel as nib
import os
import cv2
import numpy as np


def generate_all_lung_disease_data(task='lung_disease'):
    # read img and mask

    img_id = 1
    for file in os.listdir('/data/covid19_internal/covid_ct_segmentation/data/0420/Infection_Mask'):

        img_slices = nib.load('/data/covid19_internal/covid_ct_segmentation/data/0420/COVID-19-CT-Seg_20cases/%s' % file).get_fdata()

        disease_mask_slices = nib.load('/data/covid19_internal/covid_ct_segmentation/data/0420/Infection_Mask/%s' % file).get_fdata()
        lung_mask_slices = nib.load('/data/covid19_internal/covid_ct_segmentation/data/0420/Lung_Mask/%s' % file).get_fdata()

        for i in range(img_slices.shape[2]):

            # mask = disease_mask_slices[:, :, i]
            # mask_ids = np.unique(mask)

            # if mask_ids.shape[0] == 2:
            if True:

                # generate and save CT image
                img = img_slices[:, :, i]
                img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8)
                img_rgb = np.zeros((np.array(img).shape[0], np.array(img).shape[1], 3), dtype=int)
                img_rgb[:, :, 0] = img
                img_rgb[:, :, 1] = img
                img_rgb[:, :, 2] = img
                img_rgb = np.rot90(img_rgb, 1)

                os.makedirs('../data/0420/%s/images' % task, exist_ok=True)
                cv2.imwrite('../data/0420/%s/images/%d.jpg' % (task, img_id), img_rgb)

                # generate and save disease CT mask
                disease_mask = disease_mask_slices[:, :, i]
                disease_mask_rgb = np.zeros((np.array(disease_mask).shape[0], np.array(disease_mask).shape[1], 3), dtype=int)
                disease_mask_rgb[:, :, 0] = np.where(disease_mask == 1, 1, 0)
                disease_mask_rgb[:, :, 1] = np.where(disease_mask == 1, 1, 0)
                disease_mask_rgb[:, :, 2] = np.where(disease_mask == 1, 1, 0)
                disease_mask_rgb = np.rot90(disease_mask_rgb, 1)
                disease_mask_rgb = disease_mask_rgb * 255

                os.makedirs('../data/0420/%s/masks/0' % task, exist_ok=True)
                cv2.imwrite('../data/0420/%s/masks/0/%d.png' % (task, img_id), disease_mask_rgb)

                # generate and save lung CT mask
                lung_mask = lung_mask_slices[:, :, i]
                lung_mask_rgb = np.zeros((np.array(lung_mask).shape[0], np.array(lung_mask).shape[1], 3), dtype=int)
                lung_mask_rgb[:, :, 0] = np.where(lung_mask >= 1, 1, 0)
                lung_mask_rgb[:, :, 1] = np.where(lung_mask >= 1, 1, 0)
                lung_mask_rgb[:, :, 2] = np.where(lung_mask >= 1, 1, 0)
                lung_mask_rgb = np.rot90(lung_mask_rgb, 1)
                lung_mask_rgb = lung_mask_rgb * 255

                os.makedirs('../data/0420/%s/masks/1' % task, exist_ok=True)
                cv2.imwrite('../data/0420/%s/masks/1/%d.png' % (task, img_id), lung_mask_rgb)

                img_id += 1


def link_data():
    # create disease data
    os.makedirs('../data/0420/disease/masks', exist_ok=True)
    os.symlink('../lung_disease/images', '../data/0420/disease/images')
    os.symlink('../../lung_disease/masks/0', '../data/0420/disease/masks/0')

    # create disease data
    os.makedirs('../data/0420/lung/masks', exist_ok=True)
    os.symlink('../lung_disease/images', '../data/0420/lung/images')
    os.symlink('../../lung_disease/masks/1', '../data/0420/lung/masks/0')


def check_identical():
    # def is_similar(image1, image2):
    #     return image1.shape == image2.shape and not (np.bitwise_xor(image1, image2).any())

    for file in os.listdir('/data/ImageSegmentation/data/0420/lung_disease/images'):
        img1 = '/data/ImageSegmentation/data/0420/lung_disease/images/%s' % file
        img2 = '/data/ImageSegmentation/data/0420_half/lung_disease/images/%s' % file

        if open(img1, "rb").read() == open(img2, "rb").read():
            # print('image %s match' % file)
            continue
        else:
            print('image %s not match' % file)


def generate_lung_disease_data(dataset, task='lung_disease'):
    # read img and mask

    img_id = 1
    for file in os.listdir('/data/covid19_internal/covid_ct_segmentation/data/0420/Infection_Mask'):
        if dataset in file:

            img_slices = nib.load('/data/covid19_internal/covid_ct_segmentation/data/0420/COVID-19-CT-Seg_20cases/%s' % file).get_fdata()

            disease_mask_slices = nib.load('/data/covid19_internal/covid_ct_segmentation/data/0420/Infection_Mask/%s' % file).get_fdata()
            lung_mask_slices = nib.load('/data/covid19_internal/covid_ct_segmentation/data/0420/Lung_Mask/%s' % file).get_fdata()

            for i in range(img_slices.shape[2]):

                # mask = disease_mask_slices[:, :, i]
                # mask_ids = np.unique(mask)

                # if mask_ids.shape[0] == 2:
                if True:

                    # generate and save CT image
                    img = img_slices[:, :, i]
                    img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8)
                    img_rgb = np.zeros((np.array(img).shape[0], np.array(img).shape[1], 3), dtype=int)
                    img_rgb[:, :, 0] = img
                    img_rgb[:, :, 1] = img
                    img_rgb[:, :, 2] = img
                    img_rgb = np.rot90(img_rgb, 1)

                    os.makedirs('../data/0420_%s/%s/images' % task, exist_ok=True)
                    cv2.imwrite('../data/0420_%s/%s/images/%d.jpg' % (dataset, task, img_id), img_rgb)

                    # generate and save disease CT mask
                    disease_mask = disease_mask_slices[:, :, i]
                    disease_mask_rgb = np.zeros((np.array(disease_mask).shape[0], np.array(disease_mask).shape[1], 3), dtype=int)
                    disease_mask_rgb[:, :, 0] = np.where(disease_mask == 1, 1, 0)
                    disease_mask_rgb[:, :, 1] = np.where(disease_mask == 1, 1, 0)
                    disease_mask_rgb[:, :, 2] = np.where(disease_mask == 1, 1, 0)
                    disease_mask_rgb = np.rot90(disease_mask_rgb, 1)
                    disease_mask_rgb = disease_mask_rgb * 255

                    os.makedirs('../data/0420_%s/%s/masks/0' % task, exist_ok=True)
                    cv2.imwrite('../data/0420_%s/%s/masks/0/%d.png' % (dataset, task, img_id), disease_mask_rgb)

                    # generate and save lung CT mask
                    lung_mask = lung_mask_slices[:, :, i]
                    lung_mask_rgb = np.zeros((np.array(lung_mask).shape[0], np.array(lung_mask).shape[1], 3), dtype=int)
                    lung_mask_rgb[:, :, 0] = np.where(lung_mask >= 1, 1, 0)
                    lung_mask_rgb[:, :, 1] = np.where(lung_mask >= 1, 1, 0)
                    lung_mask_rgb[:, :, 2] = np.where(lung_mask >= 1, 1, 0)
                    lung_mask_rgb = np.rot90(lung_mask_rgb, 1)
                    lung_mask_rgb = lung_mask_rgb * 255

                    os.makedirs('../data/0420_%s/%s/masks/1' % task, exist_ok=True)
                    cv2.imwrite('../data/0420_%s/%s/masks/1/%d.png' % (dataset, task, img_id), lung_mask_rgb)

                    img_id += 1


# generate_lung_disease_data(dataset='radiopaedia')
# generate_all_lung_disease_data()
# link_data()

check_identical()
