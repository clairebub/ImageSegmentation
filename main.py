import os
import sys 

import argparse
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
import pytorch_lightning as pl

import archs
import data_module
import losses
import unet

ARCH_NAMES = list(archs.__dict__.keys())
LOSS_NAMES = list(losses.__dict__.keys())
LOSS_NAMES.append('BCEWithLogitsLoss')
   

def main(config): 
    train_transform = Compose([
        transforms.RandomRotate90(),
        transforms.Flip(),
        transforms.Resize(config['input_h'], config['input_w']),
    ])
    val_transform = Compose([
        transforms.Resize(config['input_h'], config['input_w']),
    ])
    dm = data_module.DataModule(config, train_transform, val_transform)
    dm.prepare_data()
    dm.setup()

    # train the nested-unet model
    model = unet.NestedUNet(num_classes=1, epochs=config['epochs'])
    trainer = pl.Trainer(
        check_val_every_n_epoch=1,
        distributed_backend='ddp', 
        gpus=config['gpus'],
        precision=16, 
#        profiler='simple', 
        max_epochs=config['epochs'])
    trainer.fit(model, dm.train_dataloader(), dm.val_dataloader())

    # testing
    #result = trainer.test(test_dataloaders=dm.test_loader())
    #print(result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None, help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=2, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=2, type=int, metavar='N', help='mini-batch size')
    parser.add_argument('--gpus', default=2, type=int, metavar='N', help='number of gpus')

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
    parser.add_argument('--sub_dataset', default='disease', help='sub_dataset name')
    parser.add_argument('--img_ext', default='.jpg', help='image file extension')
    parser.add_argument('--mask_ext', default='.png', help='mask file extension')
    parser.add_argument('--input_w', default=512, type=int, help='image width')
    parser.add_argument('--input_h', default=512, type=int, help='image height')

    config = vars(parser.parse_args())
    if config['name'] is None:
        config['name'] = '%s_%s_%s_%s_woDS' % (config['dataset'], config['sub_dataset'], config['arch'], config['loss'])
    main(config)