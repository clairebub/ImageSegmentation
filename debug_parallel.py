import argparse
import logging
import os

import matplotlib.pyplot as plt
import torch 
import torch.nn as nn 
import pytorch_lightning as pl
import torchvision

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
    

class DataModule(pl.LightningDataModule):

    def __init__(self):
        super().__init__()
        self.train_ids, self.val_ids, self.test_ids = None, None, None

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
        pass

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        raise NotImplementedError


class Model(nn.Module):

    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size, bias=True)

    def forward(self, input):
        output = self.fc(input)
        logging.info(f"In Model: device={torch.cuda.current_device()}, input.shape={input.shape}, output.shape={output.shape}")
        return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epochs', default=3, type=int, metavar='N', help='max number of epochs')
    parser.add_argument('--num_workers', default=os.cpu_count(), type=int, metavar='N', help='number of data loader workers')
    parser.add_argument('--num_gpus', default=torch.cuda.device_count(), type=int, metavar='N', help='number of gpus')
    parser.add_argument('--input_size', default=2, type=int, metavar='N')
    parser.add_argument('--output_size', default=4, type=int, metavar='N')
    parser.add_argument('--num_inputs', default=1000, type=int, metavar='N', help='number of input samples')
    parser.add_argument('-b', '--batch_size', default=16, type=int, metavar='N', help='batch size')
    parser.add_argument('--pin_memory', default=True, type=utils.str2bool)
    config = vars(parser.parse_args())

    logging.basicConfig(level=logging.INFO)
    logging.info("=" * 40)
    for k in sorted(config):
        logging.info(f'{k}={config[k]}')
    logging.info("=" * 40)

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
    if config['num_gpus'] > 0:
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


 

