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
from metrics import iou_score_from_array, dice_coef_from_array
from prediction import generate_segmentation, generate_segmented_img, generate_segmented_img_gt
from utils import AverageMeter


def two_step_test(config, test_loader, models):
    avg_meters = {'iou': AverageMeter(),
                  'dice': AverageMeter()}

    disease_model, lung_model = models

    # switch to evaluate mode
    disease_model.eval()
    lung_model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(test_loader))
        for input, target, _ in test_loader:
            input = input.cuda()
            target = target.cuda()

            # compute output
            disease_output = disease_model(input)
            lung_output = lung_model(input)

            # choose interaction of disease and lung output
            disease_output = torch.sigmoid(disease_output).data.cpu().numpy()
            disease_output = disease_output > 0.5

            lung_output = torch.sigmoid(lung_output).data.cpu().numpy()
            lung_output = lung_output > 0.5

            output = disease_output * lung_output

            # compute iou
            target = target.data.cpu().numpy()
            target = target > 0.5

            iou = iou_score_from_array(output, target)
            dice = dice_coef_from_array(output, target)

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


# def two_step_predict(config, test_loader, model):
#     # switch to evaluate mode
#     model.eval()
#
#     with torch.no_grad():
#         pbar = tqdm(total=len(test_loader))
#         for input, target, meta in test_loader:
#             input = input.cuda()
#             # target = target.cuda()
#
#             # compute output
#             if config['deep_supervision']:
#                 output = model(input)[-1]
#             else:
#                 output = model(input)
#
#             # generate segmentation
#             if config['seg_img'] == 'seg_only':
#                 generate_segmentation(config, output, meta)
#             elif config['seg_img'] == 'seg_predict':
#                 generate_segmented_img(config, input, output, meta)
#             elif config['seg_img'] == 'gt':
#                 generate_segmented_img_gt(config, input, target, meta)
#
#             pbar.update(1)
#         pbar.close()


def two_step_test_entry(config, ix_sum=10, ix=0):

    model_names = ['disease', 'lung']

    if config['name'] is None:
        if config['deep_supervision']:
            config['name'] = '%s_%s_%s_wDS' % (config['dataset'], config['sub_dataset'], config['arch'])
        else:
            config['name'] = '%s_%s_%s_woDS' % (config['dataset'], config['sub_dataset'], config['arch'])

    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)

    with open('models/%s/config.yml' % config['name'], 'w') as f:
        yaml.dump(config, f)

    cudnn.benchmark = True

    models = []
    for id, model_name in enumerate(model_names):
        # create model
        arch = config['arch'].split('_')[0]
        print("=> creating model %s" % arch)
        models.append(archs.__dict__[arch](config['num_classes'], config['input_channels'], config['deep_supervision']))

        models[id] = nn.DataParallel(models[id])

        # load trained model
        print("Reloading model 'models/%s/model.pth'..." % config['name'].replace('disease', model_names[id]))
        models[id].load_state_dict(torch.load('models/%s/model.pth' % config['name'].replace('disease', model_names[id])))

        models[id] = models[id].cuda()

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
        pass
        # two_step_predict(config, test_loader, disease_model, lung_model)
    else:
        test_result = two_step_test(config, test_loader, models)

        df = pd.DataFrame(test_result.items(), columns=["metric", "result"])
        print(df)

        # test_result_per_class = test_per_class(config, test_loader, model)
        # for class_id in range(config['num_classes']):
        #     print('[class %d]:' % class_id)
        #     df = pd.DataFrame(test_result_per_class[class_id].items(), columns=["metric", "result"])
        #     print(df)

