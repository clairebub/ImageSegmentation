import os
import torch
import cv2
import numpy as np


def generate_segmentation(config, output, meta):

    output = torch.sigmoid(output).data.cpu().numpy()

    output[output > 0.5] = 1
    output[output <= 0.5] = 0
    output = np.asarray(output, dtype="int32")

    for i in range(len(output)):

        for c in range(config['num_classes']):
            mask = cv2.normalize(output[i, c], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8)

            save_path = os.path.join('outputs', config['name'] + '_testing', str(c))
            os.makedirs(save_path, exist_ok=True)

            cv2.imwrite(os.path.join(save_path, meta['img_id'][i] + '.png'), mask)


def generate_segmented_img(config, input, output, meta):

    output = torch.sigmoid(output).data.cpu().numpy()

    output[output > 0.5] = 1
    output[output <= 0.5] = 0
    output = np.asarray(output, dtype="float32")

    input = input.data.cpu().numpy()
    input = np.asarray(input, dtype="float32")

    seg_img = input * output

    for i in range(len(output)):

        for c in range(config['num_classes']):
            mask = cv2.normalize(seg_img[i, c], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8)

            save_path = os.path.join('outputs', config['name'] + '_seg_testing', str(c))
            os.makedirs(save_path, exist_ok=True)

            cv2.imwrite(os.path.join(save_path, meta['img_id'][i] + '.jpg'), mask)


def generate_segmented_img_gt(config, input, target, meta):

    target = target.data.cpu().numpy()
    target = np.asarray(target, dtype="float32")

    input = input.data.cpu().numpy()
    input = np.asarray(input, dtype="float32")

    seg_img = input * target

    for i in range(len(target)):

        for c in range(config['num_classes']):
            mask = cv2.normalize(seg_img[i, c], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8)

            save_path = os.path.join('outputs', config['name'] + '_seg_gt', str(c))
            os.makedirs(save_path, exist_ok=True)

            cv2.imwrite(os.path.join(save_path, meta['img_id'][i] + '.jpg'), mask)


def generate_cropped_img(config, input, output, target, meta, for_training=False):

    output = torch.sigmoid(output).data.cpu().numpy()

    output[output > 0.5] = 1
    output[output <= 0.5] = 0
    output = np.asarray(output, dtype="float32")

    input = input.data.cpu().numpy()
    input = np.asarray(input, dtype="float32")

    target = target.data.cpu().numpy()
    target[target > 0] = 255
    target = np.asarray(target, dtype="float32")

    mask_index_dict = {}
    for i in range(len(output)):

        for c in range(config['num_classes']):
            img = input[i, c]
            mask = output[i, c]

            # get coordinates
            mask_index = np.where(mask == 1)

            # skip no lung images
            if len(mask_index[0]) > 0 and len(mask_index[1]) > 0:
                min_height = np.min(mask_index[0])
                max_height = np.max(mask_index[0])
                min_width = np.min(mask_index[1])
                max_width = np.max(mask_index[1])

                mask_index_dict[meta['img_id'][i]] = [min_height, max_height, min_width, max_width]

                cropped_img = img[min_height:max_height + 1, min_width:max_width + 1]

                cropped_img = cv2.normalize(cropped_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8)

                if not for_training:
                    save_path = os.path.join('outputs', config['name'] + '_crop_testing', str(c))
                    os.makedirs(save_path, exist_ok=True)

                    cv2.imwrite(os.path.join(save_path, meta['img_id'][i] + '.jpg'), cropped_img)

                else:
                    # save cropped img
                    img_path = os.path.join('data', config['dataset'] + '_cropped', config['sub_dataset'], 'images')
                    os.makedirs(img_path, exist_ok=True)
                    cv2.imwrite(os.path.join(img_path, meta['img_id'][i] + '.jpg'), cropped_img)

                    # save cropped mask
                    mask_gt = target[i, c]
                    cropped_mask_gt = mask_gt[min_height:max_height + 1, min_width:max_width + 1]

                    target_path = os.path.join('data', config['dataset'] + '_cropped', config['sub_dataset'], 'masks', str(c))
                    os.makedirs(target_path, exist_ok=True)
                    cv2.imwrite(os.path.join(target_path, meta['img_id'][i] + '.png'), cropped_mask_gt)

    return mask_index_dict
