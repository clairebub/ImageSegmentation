import argparse
import archs
import losses

from utils import str2bool
from train import train_entry
from test import test_entry
from two_step_test import two_step_test_entry

ARCH_NAMES = list(archs.__dict__.keys())
LOSS_NAMES = list(losses.__dict__.keys())
LOSS_NAMES.append('BCEWithLogitsLoss')

"""restrict GPU option"""
# find most open GPU (default use 8 gpus)
# gpu_list = get_default_gpus(4)
# gpu_ids = ','.join(map(str, gpu_list))
#
# allocate GPU
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
#
# #print("Allocated GPU %s" % gpu_ids)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None, help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=300, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=16, type=int, metavar='N', help='mini-batch size (default: 16)')

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
    parser.add_argument('--sub_dataset', default='lung', help='sub_dataset name')
    parser.add_argument('--img_ext', default='.jpg', help='image file extension')
    parser.add_argument('--mask_ext', default='.png', help='mask file extension')

    parser.add_argument('--num_classes', default=1, type=int, help='number of classes')
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

    # if config['name'] is None:
    #     if config['deep_supervision']:
    #         config['name'] = '%s_%s_%s_wDS_%s_%s' % (config['dataset'], config['sub_dataset'], config['arch'], ix, ix_sum)
    #     else:
    #         config['name'] = '%s_%s_%s_woDS_%s_%s' % (config['dataset'], config['sub_dataset'], config['arch'], ix, ix_sum)

    if config['name'] is None:
        if config['deep_supervision']:
            config['name'] = '%s_%s_%s_%s_wDS' % (config['dataset'], config['sub_dataset'], config['arch'], config['loss'])
        else:
            config['name'] = '%s_%s_%s_%s_woDS' % (config['dataset'], config['sub_dataset'], config['arch'], config['loss'])

    # iou_sc = []
    # for ix in range(num_fold):
    #     iou_sc.append(train_once(num_fold, ix))
    #
    # print('iou_sc', iou_sc, np.mean(iou_sc))

    if config['train']:
        # train
        iou = train_entry(config, ix_sum=10)
        # print(iou)
    elif config['test']:
        # test
        test_entry(config)
    elif config['two_step_test']:
        two_step_test_entry(config)
