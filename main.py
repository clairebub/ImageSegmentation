import os
import sys


import argparse

import pytorch_lightning as pl
import torch
import yaml

import archs
import data_module
import losses
import unet
import utils


ARCH_NAMES = list(archs.__dict__.keys())
LOSS_NAMES = list(losses.__dict__.keys())
LOSS_NAMES.append('BCEWithLogitsLoss')

def main(args): 
    config = vars(args)
    if config['name'] is None:
        config['name'] = 'pl_%s_%s_%s_%s_woDS' % (config['dataset'], config['sub_dataset'], config['arch'], config['loss'])
    print('=' * 40)
    for key in sorted(config):
        print('%s: %s' % (key, config[key]))
    print('=' * 40)
    os.makedirs('models/%s' % config['name'], exist_ok=True)
    with open('models/%s/config.yml' % config['name'], 'w') as f:
        yaml.dump(config, f)

    # set up the model, data module and trainer  
    model = unet.NestedUNet(config)
    dm = data_module.DataModule(config)
    dm.prepare_data()
    dm.setup()

    # run the trainer
    if(config['gpus'] != 0):
        torch.cuda.empty_cache()
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, dm.train_dataloader(), dm.val_dataloader())

    # testing
    #result = trainer.test(test_dataloaders=dm.test_loader())
    #print(result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-b', '--batch_size', default=2, type=int, metavar='N', help='mini-batch size')
    parser.add_argument('--max_epochs', default=2, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--gpus', default=torch.cuda.device_count(), type=int, metavar='N', help='number of gpus')
    parser.add_argument('--accelerator', default='dp' if torch.cuda.is_available() else None, help='dp or ddp or None')
    parser.add_argument('--precision', default=32, type=int, metavar='N', help='floating number precision')

    # train/test/predict
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--two_step_test', action='store_true')
    parser.add_argument('--predict', action='store_true')

    # model
    parser.add_argument('--name', default=None, help='model name: (default: arch+timestamp)')
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
    parser.add_argument('--input_channels', default=3, type=int, help='input channels') # 1 or 3    
    parser.add_argument('--num_inputs', default=32, type=int, help='number of inputs')    
    parser.add_argument('--num_classes', default=1, type=int, help='number of classes')
    parser.add_argument('--input_w', default=512, type=int, help='image width')
    parser.add_argument('--input_h', default=512, type=int, help='image height')
    parser.add_argument('--num_workers', default=os.cpu_count(), type=int, metavar='N', help='number of data loader workers')
    parser.add_argument('--pin_memory', default=True, type=utils.str2bool)

    args = parser.parse_args()
    main(args)