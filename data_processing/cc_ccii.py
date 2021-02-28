rom PIL import Image
import numpy as np
# from glob import glob
import os
import cv2

import torch
from torchvision import datasets, transforms

data_path = '/data/covid19_internal/covid_ct_segmentation/data/cc_ccii/ct_lesion_seg'
cc_ccii_path = '/data/covid19_internal/covid_ct_segmentation/data/cc_ccii'


def load_image(infilename):
    img = Image.open(infilename).convert('L')
    img.load()
    data = np.asarray(img, dtype="int32")

    return data


def generate_disease_data(task='disease'):
    # read img and mask

    img_id = 1
    for patient_folder in os.listdir('%s/mask/' % data_path):
        for file in os.listdir('%s/mask/%s/' % (data_path, patient_folder)):
            if 'bmp' in file or 'jpg' in file or 'JPG' in file or 'png' in file or 'PNG' in file:

                # create symlink of training images
                if not os.path.exists('../data/cc_ccii/%s/images' % task):
                    os.makedirs('../data/cc_ccii/%s/images' % task)
                os.symlink('%s/image/%s/%s' % (data_path, patient_folder, file.replace('png', 'jpg')), '../data/cc_ccii/%s/images/%d.jpg' % (task, img_id))

                # generate mask
                mask = load_image('%s/mask/%s/%s' % (data_path, patient_folder, file))

                # process segmentation label
                mask[mask == 1] = 0
                mask[mask > 1] = 1

                mask = cv2.normalize(mask, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8)

                if not os.path.exists('../data/cc_ccii/%s/masks' % task):
                    os.makedirs('../data/cc_ccii/%s/masks' % task)
                cv2.imwrite('../data/cc_ccii/%s/masks/%d.png' % (task, img_id), mask)

                img_id += 1


def generate_lung_disease_data(task='lung_disease'):
    # read img and mask

    img_id = 1
    for patient_folder in os.listdir('%s/mask/' % data_path):
        for file in os.listdir('%s/mask/%s/' % (data_path, patient_folder)):
            if 'bmp' in file or 'jpg' in file or 'JPG' in file or 'png' in file or 'PNG' in file:

                # create symlink of training images
                if not os.path.exists('../data/cc_ccii/%s/images' % task):
                    os.makedirs('../data/cc_ccii/%s/images' % task)
                os.symlink('%s/image/%s/%s' % (data_path, patient_folder, file.replace('png', 'jpg')), '../data/cc_ccii/%s/images/%d.jpg' % (task, img_id))

                ### generate disease mask
                disease_mask = load_image('%s/mask/%s/%s' % (data_path, patient_folder, file))

                # process segmentation label
                disease_mask[disease_mask <= 1] = 0
                disease_mask[disease_mask > 1] = 1

                disease_mask = cv2.normalize(disease_mask, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8)

                if not os.path.exists('../data/cc_ccii/%s/masks/0' % task):
                    os.makedirs('../data/cc_ccii/%s/masks/0' % task)
                cv2.imwrite('../data/cc_ccii/%s/masks/0/%d.png' % (task, img_id), disease_mask)

                ### generate lung disease mask
                mask = load_image('%s/mask/%s/%s' % (data_path, patient_folder, file))

                # process segmentation label
                mask[mask >= 1] = 1

                mask = cv2.normalize(mask, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8)

                if not os.path.exists('../data/cc_ccii/%s/masks/1' % task):
                    os.makedirs('../data/cc_ccii/%s/masks/1' % task)
                cv2.imwrite('../data/cc_ccii/%s/masks/1/%d.png' % (task, img_id), mask)

                img_id += 1


def read_lesion_slices_for_test():
    csv_path = '/data/covid19_internal/covid_ct_segmentation/data/cc_ccii/lesions_slices.csv'

    lesion_slices = []
    with open(csv_path) as fp:
        for line in fp:
            if line.startswith('NCP'):
                lesion_slices.append(line.strip())

    img_id = 1
    for folder1 in os.listdir('%s/NCP/' % cc_ccii_path):
        for folder2 in os.listdir('%s/NCP/%s/' % (cc_ccii_path, folder1)):

            for file in os.listdir('%s/NCP/%s/%s' % (cc_ccii_path, folder1, folder2)):
                if ('bmp' in file) or ('jpg' in file) or ('JPG' in file) or ('png' in file) or ('PNG' in file):
                    # filename, file_extension = os.path.splitext(file)

                    path = 'NCP/%s/%s/%s' % (folder1, folder2, file)

                    if path in lesion_slices:

                        # link image
                        if not os.path.exists('../data/cc_ccii/disease_testing/images'):
                            os.makedirs('../data/cc_ccii/disease_testing/images')
                        if os.path.isfile('%s/NCP/%s/%s/%s' % (cc_ccii_path, folder1, folder2, file)):
                            os.symlink('%s/NCP/%s/%s/%s' % (cc_ccii_path, folder1, folder2, file), '../data/cc_ccii/disease_testing/images/%d.jpg' % img_id)

                        # link mask (dummy mask with all black pixels)
                        if not os.path.isfile('../data/cc_ccii/black_mask.png'):
                            dummy_mask = np.zeros([512, 512, 3], dtype=np.uint8)
                            dummy_mask = cv2.normalize(dummy_mask, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8)
                            cv2.imwrite('../data/cc_ccii/black_mask.png', dummy_mask)

                        # link mask
                        if not os.path.exists('../data/cc_ccii/disease_testing/masks/0'):
                            os.makedirs('../data/cc_ccii/disease_testing/masks/0')
                        os.symlink('../../../black_mask.png', '../data/cc_ccii/disease_testing/masks/0/%d.png' % img_id)

                        img_id += 1


def read_lesion_slices_from_classification():

    for file in os.listdir('/dataImageClassification/data/cc_ccii/lesion_slices/lesion'):
        org_img = os.readlink('/data/ImageClassification/data/cc_ccii/lesion_slices/lesion/%s' % file)

        # link image
        if not os.path.exists('../data/cc_ccii/disease_testing/images'):
            os.makedirs('../data/cc_ccii/disease_testing/images')
        if os.path.isfile(org_img):
            os.symlink(org_img, '../data/cc_ccii/disease_testing/images/%s' % file)

        # link mask (dummy mask with all black pixels)
        if not os.path.isfile('../data/cc_ccii/black_mask.png'):
            dummy_mask = np.zeros([512, 512, 3], dtype=np.uint8)
            dummy_mask = cv2.normalize(dummy_mask, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8)
            cv2.imwrite('../data/cc_ccii/black_mask.png', dummy_mask)

        # link mask
        if not os.path.exists('../data/cc_ccii/disease_testing/masks/0'):
            os.makedirs('../data/cc_ccii/disease_testing/masks/0')
        os.symlink('../../../black_mask.png', '../data/cc_ccii/disease_testing/masks/0/%s' % file.replace('jpg', 'png'))


def get_data_statistics(dataset, task):
    data_path = '../data/%s/%s/images' % (dataset, task)

    os.symlink('../images', '../data/%s/%s/images/images' % (dataset, task))

    """compute image data statistics (mean, std)"""
    data_transform = transforms.Compose([transforms.Resize(size=(512, 512)),
                                         transforms.ToTensor()])

    train_data = datasets.ImageFolder(root=data_path, transform=data_transform)

    # verify number of images
    print(len(train_data))

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=4096,
        num_workers=4,
        shuffle=False
    )

    pop_mean = []
    pop_std0 = []
    pop_std1 = []
    for i, data in enumerate(train_loader, 0):
        # shape (batch_size, 3, height, width)
        images, labels = data
        numpy_image = np.asarray([item.numpy() for item in images])

        # shape (3,)
        batch_mean = np.mean(numpy_image, axis=(0, 2, 3))
        batch_std0 = np.std(numpy_image, axis=(0, 2, 3))
        batch_std1 = np.std(numpy_image, axis=(0, 2, 3), ddof=1)

        pop_mean.append(batch_mean)
        pop_std0.append(batch_std0)
        pop_std1.append(batch_std1)

    # shape (num_iterations, 3) -> (mean across 0th axis) -> shape (3,)
    pop_mean = np.array(pop_mean).mean(axis=0)
    pop_std0 = np.array(pop_std0).mean(axis=0)
    pop_std1 = np.array(pop_std1).mean(axis=0)

    print(pop_mean)
    print(pop_std0)

    os.system('unlink ../data/%s/%s/images/images' % (dataset, task))

    return pop_mean, pop_std0, pop_std1


# generate_disease_data()
# generate_lung_disease_data()
get_data_statistics('ieee8023', 'lung')

# read_lesion_slices_for_test()
# read_lesion_slices_from_classification()
