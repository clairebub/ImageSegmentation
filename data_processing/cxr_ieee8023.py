import numpy as np
import os
import cv2

img_path = '/data/covid19_internal/covid_cxr_images/real'
mask_path = '/data/covid19_internal/others/covid-chestxray-dataset/annotations/lungVAE-masks'

seg_path = '/data/ImageSegmentation/data/ieee8023'


def prepare_ieee8023_training():
    img_dict = {}
    for img in os.listdir(img_path):
        img_name = os.path.splitext(img)[0]
        img_dict[img_name] = img

    for mask in os.listdir(mask_path):
        img_name = mask.replace('_mask.png', '')

        if img_name in img_dict.keys():
            os.makedirs('%s/lung/images' % seg_path, exist_ok=True)
            os.symlink('%s/%s' % (img_path, img_dict[img_name]), '%s/lung/images/%s.jpg' % (seg_path, img_name))

            os.makedirs('%s/lung/masks/0' % seg_path, exist_ok=True)
            os.symlink('%s/%s' % (mask_path, mask), '%s/lung/masks/0/%s.png' % (seg_path, img_name))


def prepare_allcovid_testing():
    # link mask (dummy mask with all black pixels)
    if not os.path.isfile('/data/ImageSegmentation/data/black_mask.png'):
        dummy_mask = np.zeros([512, 512, 3], dtype=np.uint8)
        dummy_mask = cv2.normalize(dummy_mask, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8)
        cv2.imwrite('/data/ImageSegmentation/data/black_mask.png', dummy_mask)

    for img in os.listdir(img_path):
        img_name = os.path.splitext(img)[0]

        os.makedirs('%s/lung_real/images' % seg_path, exist_ok=True)
        os.makedirs('%s/lung_real/masks/0' % seg_path, exist_ok=True)

        try:
            os.symlink('%s/%s' % (img_path, img), '%s/lung_real/images/%s.jpg' % (seg_path, img_name))
            os.symlink('/data/ImageSegmentation/data/black_mask.png', '%s/lung_real/masks/0/%s.png' % (seg_path, img_name))
        except:
            os.unlink('%s/lung_real/images/%s.jpg' % (seg_path, img_name))
            os.unlink('%s/lung_real/masks/0/%s.png' % (seg_path, img_name))

            os.symlink('%s/%s' % (img_path, img), '%s/lung_real/images/%s.jpg' % (seg_path, img_name))
            os.symlink('/data/ImageSegmentation/data/black_mask.png', '%s/lung_real/masks/0/%s.png' % (seg_path, img_name))


def prepare_all_pruning():
    classes = ['COVID19', 'NORMAL', 'PNEUMONIA']
    cxr_covid19_all_path = '/data/ImageClassification/data/cxr_covid19_all'

    for c in classes:
        for img in os.listdir('%s/%s' % (cxr_covid19_all_path, c)):
            img_name = os.path.splitext(img)[0]

            os.rename('%s/%s/%s' % (cxr_covid19_all_path, c, img), '%s/%s/%s.jpg' % (cxr_covid19_all_path, c, img_name))

            os.makedirs('%s/lung_%s/images' % (seg_path, c), exist_ok=True)
            os.makedirs('%s/lung_%s/masks/0' % (seg_path, c), exist_ok=True)

            os.symlink('%s/%s/%s.jpg' % (cxr_covid19_all_path, c, img_name), '%s/lung_%s/images/%s.jpg' % (seg_path, c, img_name))
            os.symlink('/data/ImageSegmentation/data/black_mask.png', '%s/lung_%s/masks/0/%s.png' % (seg_path, c, img_name))


# prepare_ieee8023_training()
# prepare_allcovid_testing()
prepare_all_pruning()
