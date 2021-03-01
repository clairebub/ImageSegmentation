from glob import glob
import os

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split

from dataset import Dataset

class DataModule(pl.LightningDataModule):

    def __init__(self, config, train_transform=None, val_transform=None, test_transform=None):
        super().__init__()
        self.img_ext = '.jpg'
        self.mask_ext = '.png'
        self.num_classes = 1
        self.num_workers = 4 * config['gpus']
        self.data_dir = os.path.join('data', config['dataset'], config['sub_dataset'])
        self.batch_size = config['batch_size']
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.train_ids, self.val_ids, self.test_ids = None, None, None

    def prepare_data(self):
        # This method is used to define the processes that are meant to be 
        # performed by only one GPU. It’s usually used to handle the task of
        # downloading the data. 
        pass

    def setup(self, stage=None):
        # This method is used to define the process that is meant to be performed by all the available GPU. 
        # It’s usually used to handle the task of loading the data. 
        img_files = glob(os.path.join(self.data_dir, 'images', '*.jpg'))
        img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_files]
        train_size = int(len(img_ids) * 0.9)
        val_size = len(img_ids) - train_size
        self.train_ids, self.val_ids = random_split(
            img_ids, [train_size, val_size], generator=torch.Generator().manual_seed(41))

    def train_dataloader(self):
        return self._create_dataloader(self.train_ids, True, False, self.train_transform)

    def val_dataloader(self):
        return self._create_dataloader(self.val_ids, False, True, self.val_transform)

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
            pin_memory=True,
            shuffle=shuffle,
            drop_last=drop_last)

if __name__ == '__main__':
    data_dir = 'data/0420/lung'
    batch_size = 4
    data_module = DataModule(data_dir, batch_size)
    data_module.prepare_data()
    data_module.setup()
    data_module.train_dataloader()
