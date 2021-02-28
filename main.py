import argparse
import sys 

from albumentations.augmentations import transforms
from albumentations.core.composition import Compose

import archs
import losses

ARCH_NAMES = list(archs.__dict__.keys())
LOSS_NAMES = list(losses.__dict__.keys())
LOSS_NAMES.append('BCEWithLogitsLoss')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None, help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=2, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=2, type=int, metavar='N', help='mini-batch size')

    # train/test
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--two_step_test', action='store_true')
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--seg_img', default='seg_img', choices=['seg_only', 'seg_img', 'gt', 'crop_img'])


    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default='NestedUNet',
                        help='model architecture: ' +
                        ' | '.join(ARCH_NAMES) +
                        ' (default: NestedUNet)')
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
    parser.add_argument('--input_w', default=512, type=int, help='image width')
    parser.add_argument('--input_h', default=512, type=int, help='image height')

    config = parser.parse_args()

    return config


if __name__ == '__main__':
    config = vars(parse_args())

    data_dir = os.path.join('data', config['dataset'], config['sub_dataset']) 
    train_transform = Compose([
        transforms.RandomRotate90(),
        transforms.Flip(),
        transforms.Resize(config['input_h'], config['input_w']),
    ])
    val_transform = Compose([
        transforms.Resize(config['input_h'], config['input_w']),
    ])
    data_module = DataModule(data_dir, config['batch_size'], train_transform, val_transform)
    data_module.prepare_data()
    data_module.setup()

    # train the nested-unet model
    model = NestedUNet()
    trainer = pl.Trainer()
    trainer.fit(model, data_module.train_dataloader())