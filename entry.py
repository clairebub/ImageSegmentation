import argparse

import torch
import torch.multiprocessing as mp
import torch.distributed as dist

import archs
import losses
from test import test_entry
from train import train_entry, demo_basic
from two_step_test import two_step_test_entry
from utils import str2bool

ARCH_NAMES = list(archs.__dict__.keys())
LOSS_NAMES = list(losses.__dict__.keys())
LOSS_NAMES.append('BCEWithLogitsLoss')

             
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None, help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=1, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=16, type=int, metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('--accelerator', default='dp' if torch.cuda.is_available() else None, help='dp or ddp or None')
    parser.add_argument('--world_size', default=torch.cuda.device_count(), help='dp or ddp or None')
    # train/test
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--two_step_test', action='store_true')
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--seg_img', default='seg_img', choices=['seg_only', 'seg_img', 'gt', 'crop_img'])

    # data
    parser.add_argument('--save_split', action='store_true')
    parser.add_argument('--load_split', action='store_true')

    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default='NestedUNet',
                        help='model architecture: ' +
                        ' | '.join(ARCH_NAMES) +
                        ' (default: NestedUNet)')
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int, help='input channels') # 1 or 3
    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=LOSS_NAMES,
                        help='loss: ' +
                        ' | '.join(LOSS_NAMES) +
                        ' (default: BCEDiceLoss)')

    # dataset
    parser.add_argument('--dataset', default='0420', help='dataset name')
    parser.add_argument('--sub_dataset', default='disease', help='sub_dataset name')
    parser.add_argument('--img_ext', default='.jpg', help='image file extension')
    parser.add_argument('--mask_ext', default='.png', help='mask file extension')
    parser.add_argument('--num_classes', default=1, type=int, help='number of classes')
    parser.add_argument('--num_inputs', default=0, type=int, help='number of inputs')
    parser.add_argument('--input_w', default=512, type=int, help='image width')
    parser.add_argument('--input_h', default=512, type=int, help='image height')

    # optimizer
    parser.add_argument('--optimizer', default='SGD',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                        ' | '.join(['Adam', 'SGD']) +
                        ' (default: Adam)')
    parser.add_argument('--lr', '--learning_rate', default=1e-3, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool, help='nesterov')

    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'CosineAnnealingWarmRestarts', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float, help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=10, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2/3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int, metavar='N', help='early stopping (default: -1)')   
    parser.add_argument('--num_workers', default=4, type=int)

    config = parser.parse_args()

    return config


if __name__ == '__main__':
    config = vars(parse_args())

    if config['name'] is None:
        if config['deep_supervision']:
            config['name'] = '%s_%s_%s_%s_wDS' % (config['dataset'], config['sub_dataset'], config['arch'], config['loss'])
        else:
            config['name'] = '%s_%s_%s_%s_woDS' % (config['dataset'], config['sub_dataset'], config['arch'], config['loss'])
    
    print('-' * 20)
    for key in sorted(config):
        print('%s: %s' % (key, config[key]))
    print('-' * 20)

    if config['train']:
        # train
        if config['accelerator'] == 'ddp':
            world_size = config['world_size']
            print('spawn train_entry for ddp')
            mp.spawn(demo_basic,
                     args=(world_size,),
                     nprocs=world_size,
                     join=True)
        else:
            train_entry(config)
    elif config['test']:
        # test
        test_entry(config)
    elif config['two_step_test']:
        two_step_test_entry(config)
