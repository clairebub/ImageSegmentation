import argparse
import logging
import os

import matplotlib.pyplot as plt
import torch 
import torch.nn as nn 
import torch.optim as optim
from torch.optim import lr_scheduler
import pytorch_lightning as pl
import torchvision

import data_module
import utils


class RandomDataset(torch.utils.data.Dataset):

    def __init__(self, input_size, num_inputs):
        self.len = num_inputs
        self.data = torch.randn(num_inputs, input_size)

    def __getitem__(self, index):
        logging.debug(f'data_loader worker info: {torch.utils.data.get_worker_info()}')
        return self.data[index]

    def __len__(self):
        return self.len
    

class RandomDataset2(torch.utils.data.Dataset):

    def __init__(self, input_size, output_size, num_inputs):
        self.len = num_inputs
        self.x = torch.randn(num_inputs, input_size)
        self.y = torch.randn(num_inputs, output_size)

    def __getitem__(self, index):
        logging.debug(f'data_loader worker info: {torch.utils.data.get_worker_info()}')
        return self.x[index], self.y[index], index

    def __len__(self):
        return self.len


class Model(nn.Module):

    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size, bias=True)

    def forward(self, input):
        output = self.fc(input)
        logging.info(f"In Model: device={torch.cuda.current_device() if torch.cuda.is_available() else 0}, input.shape={input.shape}, output.shape={output.shape}")
        return output


class DataModule(pl.LightningDataModule):

    def __init__(self, config):
        super().__init__()
        self.config = config

    def prepare_data(self):
        # This method is used to define the processes that are meant to be 
        # performed by only one GPU. It’s usually used to handle the task of
        # downloading the data. 
        pass

    def setup(self, stage=None):
        # This method is used to define the process that is meant to be performed by all the available GPU. 
        # It’s usually used to handle the task of loading the data. 
        pass

    def train_dataloader(self):
        dataset = RandomDataset2(
            config['input_size'],
            config['output_size'],
            config['num_inputs'])
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config['batch_size'],
            num_workers=self.config['num_workers'],
            pin_memory=self.config['pin_memory'],
            shuffle=True, 
            drop_last=True)

    def val_dataloader(self):
        raise NotImplementedError

    def test_dataloader(self):
        raise NotImplementedError


class Model2(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.net = Model(config['input_size'], config['output_size'])
        self.criterion = nn.MSELoss()

    def forward(self, input):
        output = self.net(input)
        logging.info(f"In Model2: device={torch.cuda.current_device() if torch.cuda.is_available() else 0}, input.shape={input.shape}, output.shape={output.shape}")
        return output

    def training_step(self, batch, batch_idx):
        logging.info(f"In training_step(): device={torch.cuda.current_device() if torch.cuda.is_available() else 0 }, batch={batch_idx}, batch.length={len(batch)}")
        x, y, _ = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        return loss 

    def training_step_end(self, batch_parts):
        logging.info(f"In training_step_end(): device={torch.cuda.current_device() if torch.cuda.is_available() else 0}, batch_parts={batch_parts}")   
        loss = batch_parts
        if len(loss.shape) > 0:
            loss = torch.mean(loss)
        logging.info(f"In training_step_end(): device={torch.cuda.current_device() if torch.cuda.is_available() else 0}, loss={loss}")   
        return loss

    def configure_optimizers(self):
        lr = 1e-3
        momentum = 0.9
        weight_decay = 1e-4
        params = filter(lambda p: p.requires_grad, self.parameters())
        sgd = optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        scheduler = lr_scheduler.CosineAnnealingLR(sgd, self.config['max_epochs'])
        return [sgd], [scheduler]


def debug_lighting(args):
    config = vars(args)
    model = Model2(config)
    dm = DataModule(config)
    dm.prepare_data()
    dm.setup()
    trainer = pl.Trainer.from_argparse_args(args)

    # run the trainer
    if(config['gpus'] != 0):
        torch.cuda.empty_cache()
    trainer.fit(model, dm.train_dataloader())
    pass 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpus', default=torch.cuda.device_count(), type=int, metavar='N', help='number of gpus')
    parser.add_argument('--accelerator', default='dp' if torch.cuda.is_available() else None, help='dp or ddp or None')
    parser.add_argument('--max_epochs', default=3, type=int, metavar='N', help='max number of epochs')
    parser.add_argument('--num_workers', default=os.cpu_count(), type=int, metavar='N', help='number of data loader workers')
    parser.add_argument('--input_size', default=2, type=int, metavar='N')
    parser.add_argument('--output_size', default=4, type=int, metavar='N')
    parser.add_argument('--num_inputs', default=1000, type=int, metavar='N', help='number of input samples')
    parser.add_argument('-b', '--batch_size', default=16, type=int, metavar='N', help='batch size')
    parser.add_argument('--pin_memory', default=True, type=utils.str2bool)
    parser.add_argument('--input_w', default=512, type=int, help='image width')
    parser.add_argument('--input_h', default=512, type=int, help='image height')
    
    args = parser.parse_args()
    config = vars(args)

    logging.basicConfig(level=logging.INFO)
    logging.info("=" * 40)
    for k in sorted(config):
        logging.info(f'{k}={config[k]}')
    logging.info("=" * 40)

    debug_lighting(args)
    import sys
    sys.exit(1)

    targets = torch.rand(config['batch_size'], config['output_size'])
    dataset = RandomDataset(config['input_size'], config['num_inputs'])
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
        shuffle=True, 
        drop_last=True)
    model = Model(config['input_size'], config['output_size'])
    model = nn.DataParallel(model)
    if config['gpus'] > 0:
        model = model.cuda()
        targets = targets.cuda()
   
    optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    criterion = nn.MSELoss()
    # # Print model's state_dict
    # print("Model's state_dict:")
    # for param_tensor in model.state_dict():
    #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    # print()
    # # Print optimizer's state_dict
    # print("Optimizer's state_dict:")
    # for var_name in optim.state_dict():
    #     print(var_name, "\t", optim.state_dict()[var_name])
    # print()

    for epoch in range(config['max_epochs']):
        logging.info(f"epoch={epoch}")
        for inputs in data_loader:
            logging.info(f'batch inputs.shape={inputs.shape}')
            inputs = inputs.cuda()
            outputs = model(inputs)
            logging.info(f'batch outputs.shape={outputs.shape}')
            loss = criterion(outputs, targets)
            model.zero_grad()
            loss.backward()
            optim.step()
    # model = torchvision.models.resnet18(pretrained=True)
    # num_inputs = 100
    # labels = torch.rand(num_inputs, 1000)
    # x = torch.rand(num_inputs, 3, 64, 64)
    # y_hat = model2(x)
    # loss = (y_hat - labels)
    # loss = loss.sum()
    # loss.backward()


 

