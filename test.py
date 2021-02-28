import os
import json
import pickle
from collections import OrderedDict
from glob import glob

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf
from tqdm import tqdm

import archs
from dataset import Dataset
from metrics import iou_score, dice_coef
from prediction import generate_segmentation, generate_segmented_img, generate_segmented_img_gt, generate_cropped_img
from utils import AverageMeter


def test(config, test_loader, model):
    avg_meters = {'iou': AverageMeter(),
                  'dice': AverageMeter()}

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(test_loader))
        for input, target, meta in test_loader:
            input = input.cuda()
            target = target.cuda()

            # compute output
            if config['deep_supervision']:
                output = model(input)[-1]
            else:
                output = model(input)

            iou = iou_score(output, target)
            dice = dice_coef(output, target)

            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))

            postfix = OrderedDict([
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg)
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([
        ('iou', avg_meters['iou'].avg),
        ('dice', avg_meters['dice'].avg)
    ])


def test_per_class(config, test_loader, model):
    avg_meters = []
    for _ in range(config['num_classes']):
        avg_meters.append({'iou': AverageMeter(), 'dice': AverageMeter()})

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for input, target, _ in test_loader:
            input = input.cuda()
            target = target.cuda()

            # compute output
            if config['deep_supervision']:
                output = model(input)[-1]
            else:
                output = model(input)

            for class_id in range(output.shape[1]):
                output_per_class = torch.unsqueeze(output[:, class_id, :, :], 1)
                target_per_class = torch.unsqueeze(target[:, class_id, :, :], 1)

                iou = iou_score(output_per_class, target_per_class)
                dice = dice_coef(output_per_class, target_per_class)

                avg_meters[class_id]['iou'].update(iou, input.size(0))
                avg_meters[class_id]['dice'].update(dice, input.size(0))

    results = []
    for class_id in range(config['num_classes']):
        results.append(
            OrderedDict([
                ('iou', avg_meters[class_id]['iou'].avg),
                ('dice', avg_meters[class_id]['dice'].avg)
            ])
        )

    return results


def predict(config, test_loader, model):
    # switch to evaluate mode
    model.eval()

    mask_index_dict = {}
    with torch.no_grad():
        pbar = tqdm(total=len(test_loader))
        for input, target, meta in test_loader:
            input = input.cuda()
            # target = target.cuda()

            # compute output
            if config['deep_supervision']:
                output = model(input)[-1]
            else:
                output = model(input)

            # generate segmentation
            if config['seg_img'] == 'seg_only':
                generate_segmentation(config, output, meta)
            elif config['seg_img'] == 'seg_img':
                generate_segmented_img(config, input, output, meta)
            elif config['seg_img'] == 'gt':
                generate_segmented_img_gt(config, input, target, meta)
            elif config['seg_img'] == 'crop_img':
                batch_mask_index_dict = generate_cropped_img(config, input, output, target, meta)
                mask_index_dict.update(batch_mask_index_dict)

            pbar.update(1)
        pbar.close()

    if config['seg_img'] == 'crop_img':
        mask_index_file = open(os.path.join('outputs', config['name'] + '_crop_testing', 'mask_index.p'), 'wb')
        pickle.dump(mask_index_dict, mask_index_file)


def test_entry(config, ix_sum=10, ix=0):

    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)

    with open('models/%s/config.yml' % config['name'], 'w') as f:
        yaml.dump(config, f)

    cudnn.benchmark = True

    # create model
    arch = config['arch'].split('_')[0]
    print("=> creating model %s" % arch)
    model = archs.__dict__[arch](config['num_classes'],
                                 config['input_channels'],
                                 config['deep_supervision'])

    model = nn.DataParallel(model)

    # load trained model
    print("Reloading model 'models/%s/model.pth'..." % config['name'])
    model.load_state_dict(torch.load('models/%s/model.pth' % config['name']))

    model = model.cuda()

    # Data loading code
    data_path = os.path.join('data', config['dataset'])
    img_ids = glob(os.path.join(data_path, config['sub_dataset'] + '_testing', 'images', '*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    split_filename = os.path.join(data_path, 'split.p')
    if config['load_split'] is True:
        split_file = open(split_filename, 'rb')
        _, val_img_ids = pickle.load(split_file)
    else:
        val_img_ids = img_ids

    # read statistics
    with open('data_processing/data_statistics.json') as json_file:
        data = json.load(json_file)

        entry = '%s_%s' % (config['dataset'], config['sub_dataset'])
        mean = data[entry]['mean']
        std = data[entry]['std']

        print("%s: mean=[%s] std=[%s]" % (config['dataset'], ', '.join(map(str, mean)), ', '.join(map(str, std))))

    # data normalization
    normalize = transforms.Normalize(mean=mean, std=std)

    test_transform = Compose([
        transforms.Resize(config['input_h'], config['input_w']),
        # normalize,
    ])

    test_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('data', config['dataset'], config['sub_dataset'] + '_testing', 'images'),
        mask_dir=os.path.join('data', config['dataset'], config['sub_dataset'] + '_testing', 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=test_transform)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    if config['predict']:
        predict(config, test_loader, model)
    else:
        test_result = test(config, test_loader, model)

        df = pd.DataFrame(test_result.items(), columns=["metric", "result"])
        print(df)

        test_result_per_class = test_per_class(config, test_loader, model)
        for class_id in range(config['num_classes']):
            print('[class %d]:' % class_id)
            df = pd.DataFrame(test_result_per_class[class_id].items(), columns=["metric", "result"])
            print(df)

    # test_result = test(config, test_loader, model)
    #
    # df = pd.DataFrame(test_result.items(), columns=["metric", "result"])
    # print(df)
