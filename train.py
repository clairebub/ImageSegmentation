import os
import json
import pickle
from collections import OrderedDict
from glob import glob

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm

import archs
import losses
from dataset import Dataset
from metrics import iou_score, dice_coef
from utils import AverageMeter
from test import test_per_class


def train(config, train_loader, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'dice': AverageMeter()}

    model.train()

    pbar = tqdm(total=len(train_loader))
    for input, target, _ in train_loader:
        # input --> bz * channel(3) * h * w
        # target --> bz * 1 * h * w
        # print ('---', input.size())
        #input = input.cuda()
        #target = target.cuda()

        # compute output
        if config['deep_supervision']:
            outputs = model(input)
            loss = 0
            for output in outputs:
                loss += criterion(input, output, target)
            loss /= len(outputs)

            output = outputs[-1]
        else:
            output = model(input)

            loss = criterion(input, output, target)

        # loss = loss * 2

        #     if config['weight_loss'] is True:
        #         disease_loss = torch.mean(target[:, 0, :, :] * criterion(output[:, 0, :, :], target[:, 0, :, :]))
        #         ratio = torch.sum(1 - target[:, 0, :, :]) / torch.sum(target[:, 0, :, :])
        #
        #         loss = loss + (ratio - 1) * disease_loss
        #
        #     # loss = criterion(output, target)
        #
        # # add lung loss based on lung segmentation gt
        # if config['lung_loss'] is True:
        #     output = torch.sigmoid(output)
        #
        #     disease_output = output[:, 0, :, :]
        #     # lung_output = output[:, 1, :, :]
        #
        #     disease_output = disease_output > 0.5
        #     # lung_output = lung_output > 0.5
        #
        #     lung_target = target[:, 1, :, :]
        #     lung_loss_tensor = disease_output.float() - lung_target.float()
        #
        #     # diff = disease_output_.float() - lung_output_.float()
        #     lung_loss_tensor[lung_loss_tensor < 0] = 0
        #     lung_regularizer = torch.mean(lung_loss_tensor)
        #
        #     # coeff = torch.log10(loss / lung_regularizer)
        #     # loss = loss + coeff * lung_regularizer / 2
        #
        #     loss = loss + 0.2 * lung_regularizer
        #
        # # add color loss based on covid prior knowledge - lighter contrast
        # if config['color_loss'] is True:
        #     # color_loss_tensor = target[:, 0, :, :] * input[:, 0, :, :] * criterion(output[:, 0, :, :], target[:, 0, :, :])
        #
        #     disease_output = output[:, 0, :, :]
        #     inverse_disease_output = disease_output < 0.5
        #     inverse_disease_output = inverse_disease_output.float()
        #
        #     color_loss_tensor = target[:, 0, :, :] * input[:, 0, :, :] * inverse_disease_output
        #
        #     color_regularizer = torch.mean(color_loss_tensor)
        #
        #     loss = loss + 0.6 * color_regularizer

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iou = iou_score(output, target)
        dice = dice_coef(output, target)

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))
        avg_meters['dice'].update(dice, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
            ('dice', avg_meters['dice'].avg)
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([
        ('loss', avg_meters['loss'].avg),
        ('iou', avg_meters['iou'].avg),
        ('dice', avg_meters['dice'].avg)
    ])


def validate(config, val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'dice': AverageMeter()}

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            #input = input.cuda()
            #target = target.cuda()

            # compute output

            if config['deep_supervision']:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(input, output, target)
                loss /= len(outputs)

                iou = iou_score(outputs[-1], target)
                dice = dice_coef(outputs[-1], target)
            else:
                output = model(input)
                loss = criterion(input, output, target)

                iou = iou_score(output, target)
                dice = dice_coef(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg)
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([
        ('loss', avg_meters['loss'].avg),
        ('iou', avg_meters['iou'].avg),
        ('dice', avg_meters['dice'].avg)
    ])


def train_entry(config, ix_sum=10, ix=0):

    os.makedirs('models/%s' % config['name'], exist_ok=True)

    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)

    with open('models/%s/config.yml' % config['name'], 'w') as f:
        yaml.dump(config, f)

    # define loss function (criterion)
    if config['loss'] == 'BCEWithLogitsLoss':
        #criterion = nn.BCEWithLogitsLoss().cuda()
        criterion = nn.BCEWithLogitsLoss()
    else:
        #criterion = losses.__dict__[config['loss']]().cuda()
        criterion = losses.__dict__[config['loss']]()

    cudnn.benchmark = True

    # create model
    arch = config['arch'].split('_')[0]

    print("=> creating model %s" % arch)
    model = archs.__dict__[arch](config['num_classes'],
                                 config['input_channels'],
                                 config['deep_supervision'])

    model = nn.DataParallel(model)
    #model = model.cuda()

    # load trained model
    model_file = 'models/%s/model.pth' % config['name']
    if os.path.isfile(model_file):
        print("Reloading model ...")
        model.load_state_dict(torch.load(model_file))

    params = filter(lambda p: p.requires_grad, model.parameters())
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(
            params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],
                              nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50)
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=config['factor'], patience=config['patience'], min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')], gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError

    # Data loading code
    data_path = os.path.join('data', config['dataset'])
    img_ids = glob(os.path.join(data_path, config['sub_dataset'], 'images', '*' + config['img_ext']))
    # img_ids = [os.path.basename(p) for p in img_ids]
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
    train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)

    split_filename = os.path.join(data_path, 'split.p')
    if config['save_split'] is True:
        split_file = open(split_filename, 'wb')
        pickle.dump((train_img_ids, val_img_ids), split_file)

    if config['load_split'] is True:
        split_file = open(split_filename, 'rb')
        train_img_ids, val_img_ids = pickle.load(split_file)

    # 5 fold validation

    # img_ids = os.listdir(os.path.join('inputs', config['dataset'], 'images'))
    # data_src = list(set([e.split('-')[0] for e in img_ids]))
    # print (data_src)
    # np.random.shuffle(data_src)
    # val_size = int(len(data_src) * 0.2)
    # train_src, val_src = data_src[:train_size], data_src[train_size:]
    # train_img_ids = [e.split('.')[0] for e in img_ids if e.split('-')[0] in train_src]
    # val_img_ids = [e.split('.')[0] for e in img_ids if e.split('-')[0] in val_src]

    '''
    Since it is gray image, some augmentation methods not used
    '''

    # read statistics
    with open('data_processing/data_statistics.json') as json_file:
        data = json.load(json_file)

        entry = '%s_%s' % (config['dataset'], config['sub_dataset'])
        mean = data[entry]['mean']
        std = data[entry]['std']

        print("%s: mean=[%s] std=[%s]" % (config['dataset'], ', '.join(map(str, mean)), ', '.join(map(str, std))))

    # data normalization
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = Compose([
        transforms.RandomRotate90(),
        transforms.Flip(),
        # transforms.HorizontalFlip(),
        # transforms.Rotate(limit=5),
        # OneOf([
        #     transforms.HueSaturationValue(),
        #     transforms.RandomBrightnessContrast(),
        #     # transforms.RandomContrast(),
        # ], p=1),
        transforms.Resize(config['input_h'], config['input_w']),
        # normalize,
    ])

    # train_transform = Compose([
    #     # transforms.RandomBrightnessContrast(0.05),
    #     # transforms.HueSaturationValue(),
    #     transforms.Rotate(5),
    #     # transforms.Resize(int(config['input_h'] * 1.1), int(config['input_w'] * 1.1)),
    #     # transforms.RandomCrop(config['input_h'], config['input_w']),
    #     transforms.Resize(config['input_h'], config['input_w']),
    #     normalize,
    # ])

    val_transform = Compose([
        transforms.Resize(config['input_h'], config['input_w']),
        # normalize,
    ])

    # split_len = len(img_ids) // ix_sum
    # val_img_ids = img_ids[ix * split_len: (ix + 1) * split_len]
    # train_img_ids = img_ids[:ix * split_len] + img_ids[(ix + 1) * split_len:]

    train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=os.path.join('data', config['dataset'], config['sub_dataset'], 'images'),
        mask_dir=os.path.join('data', config['dataset'], config['sub_dataset'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=train_transform)
    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('data', config['dataset'], config['sub_dataset'], 'images'),
        mask_dir=os.path.join('data', config['dataset'], config['sub_dataset'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('iou', []),
        ('dice', []),
        ('val_loss', []),
        ('val_iou', []),
        ('val_dice', [])
    ])

    log_per_class = []
    for _ in range(config['num_classes']):
        log_per_class.append(OrderedDict([('val_iou', []), ('val_dice', [])]))

    best_iou = 0
    best_iou_per_class = [0] * config['num_classes']

    trigger = 0
    for epoch in range(config['epochs']):
        print('Epoch [%d/%d]' % (epoch, config['epochs']))

        # train for one epoch
        train_log = train(config, train_loader, model, criterion, optimizer)
        # evaluate on validation set
        val_log = validate(config, val_loader, model, criterion)

        # evaluate on validation set for each class
        val_per_class_log = test_per_class(config, val_loader, model)

        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])

        print('loss %.4f - iou %.4f - dice %.4f - val_loss %.4f - val_iou %.4f - val_dice %.4f'
              % (train_log['loss'], train_log['iou'], train_log['dice'],
                 val_log['loss'], val_log['iou'], val_log['dice']))

        log['epoch'].append(epoch)
        log['lr'].append(config['lr'])

        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['dice'].append(train_log['dice'])

        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        log['val_dice'].append(val_log['dice'])

        pd.DataFrame(log).to_csv('models/%s/log.csv' %
                                 config['name'], index=False)

        # write into log for each class
        for class_id in range(config['num_classes']):
            log_per_class[class_id]['val_iou'].append(val_per_class_log[class_id]['iou'])
            log_per_class[class_id]['val_dice'].append(val_per_class_log[class_id]['dice'])

            print('[class %d] val_iou %.4f - val_dice %.4f'
                  % (class_id, val_per_class_log[class_id]['iou'], val_per_class_log[class_id]['dice']))

            pd.DataFrame(log_per_class[class_id]).to_csv('models/%s/log_%d.csv' %
                                                         (config['name'], class_id), index=False)

            if val_per_class_log[class_id]['iou'] > best_iou_per_class[class_id]:
                torch.save(model.state_dict(), 'models/%s/model_%d.pth' % (config['name'], class_id))
                best_iou_per_class[class_id] = val_per_class_log[class_id]['iou']
                print("===> saved best model for class %d" % class_id)

        trigger += 1

        # torch.save(model.state_dict(), 'models/%s/model.pth' % config['name'])

        if val_log['iou'] > best_iou:
            torch.save(model.state_dict(), 'models/%s/model.pth' %
                       config['name'])
            best_iou = val_log['iou']
            print("=> saved best model")
            trigger = 0

        # early stopping
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break

        #torch.cuda.empty_cache()
        torch.cuda.empty_cache()

        print("\n[Best Results:]")
        print('- [Overall IoU] %.4f' % best_iou)
        for class_id in range(config['num_classes']):
            print('-- [class %d IoU] %.4f' % (class_id, best_iou_per_class[class_id]))

    return best_iou
