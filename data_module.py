from glob import glob
import os

from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split

from dataset import Dataset


class DataModule(pl.LightningDataModule):

    def __init__(self, config):
        train_transform = Compose([
            transforms.RandomRotate90(),
            transforms.Flip(),
            transforms.Resize(config['input_h'], config['input_w']),
        ])
        val_transform = Compose([
            transforms.Resize(config['input_h'], config['input_w']),
        ])

        super().__init__()
        self.img_ext = config['img_ext']
        self.mask_ext = config['mask_ext']
        self.num_classes = config['num_classes']
        self.num_workers = config['num_workers']
        self.data_dir = os.path.join('data', config['dataset'], config['sub_dataset'])
        self.batch_size = config['batch_size']
        self.pin_memory = config['pin_memory']
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = None
        self.train_ids, self.val_ids, self.test_ids = None, None, None
        self.num_inputs = config['num_inputs']

    def prepare_data(self):
        # This method is used to define the processes that are meant to be 
        # performed by only one GPU. It’s usually used to handle the task of
        # downloading the data. 
        pass

    def setup(self, stage=None):
        # This method is used to define the process that is meant to be performed by all the available GPU. 
        # It’s usually used to handle the task of loading the data. 
        img_files = glob(os.path.join(self.data_dir, 'images', '*'+self.img_ext))
        img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_files]
        if self.num_inputs > 0:
            img_ids = img_ids[:self.num_inputs]
        train_size = int(len(img_ids) * 0.9)
        val_size = len(img_ids) - train_size
        self.train_ids, self.val_ids = random_split(
            img_ids, [train_size, val_size], generator=torch.Generator().manual_seed(41))

    def train_dataloader(self):
        shuffle = True
        drop_last = True
        return self._create_dataloader(self.train_ids, shuffle, drop_last, self.train_transform)

    def val_dataloader(self):
        shuffle = False
        drop_last = True
        return self._create_dataloader(self.val_ids, shuffle, drop_last, self.val_transform)

    def test_dataloader(self):
        raise NotImplementedError

    def _create_dataloader(self, img_ids, shuffle, drop_last, transform):
        
        dataset = Dataset(
            img_ids,
            os.path.join(self.data_dir, 'images'),
            os.path.join(self.data_dir, 'masks'),
            self.img_ext,
            self.mask_ext,
            self.num_classes, 
            transform)

        return torch.utils.data.DataLoader(
            dataset,
            self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=shuffle,
            drop_last=drop_last)


if __name__ == '__main__':

    data_dir = 'data/0420/lung'
    batch_size = 4
    config = {
        'img_ext': '.jpg', 
        'mask_ext': '.png', 
        'input_h': 512,
        'input_w': 512,
        'num_classes': 1, 
        'num_workers': 2,
        'dataset': '0420', 
        'sub_dataset': 'disease', 
        'batch_size': 16, 
        'pin_memory': True}
    data_module = DataModule(config)
    data_module.prepare_data()
    data_module.setup()
    for x in data_module.train_dataloader():
        print(x)
