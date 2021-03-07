import logging
import os

import pytorch_lightning as pl
from pytorch_lightning import Trainer
import torch
from torch import optim, nn
from torch.optim import lr_scheduler
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split

from losses import BCEDiceLoss, BCEDiceLungLoss, WeightedBCEDiceLoss
from metrics import iou_score, dice_coef
from utils import AverageMeter

logging.basicConfig(level=logging.DEBUG)

class VGGBlock(nn.Module):

    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class NestedUNet(pl.LightningModule):

    def __init__(self, config, **kwargs):
        super().__init__()
        self.avg_meters = {
            'loss': AverageMeter(),
            'iou': AverageMeter(),
            'dice': AverageMeter()}
        self.criterion = BCEDiceLoss()
        self.max_epochs = config['max_epochs']

        # load trained model
        self.model_file = 'models/%s/model.pth' % config['name']
        if os.path.isfile(self.model_file):
            print("Reloading model ...")
            self.load_state_dict(torch.load(self.model_file))


        nb_filter = [32, 64, 128, 256, 512]
        num_classes = config['num_classes']
        input_channels = config['input_channels']

        self.deep_supervision = False

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2] * 2 + nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1] * 3 + nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0] * 4 + nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        output = self.final(x0_4)
        logging.debug(f"In forward(): device={torch.cuda.current_device()}, input.shape={input.shape}, output.shape={output.shape}")
        return output

    def training_step(self, batch, batch_idx):
        logging.debug(f"In training_step(): device={torch.cuda.current_device()}, batch={batch_idx}")
        x, y, _ = batch
        y_hat = self(x)
        #return self.criterion(x, y_hat, y)
        return x, y, y_hat

    def training_step_end(self, batch_parts):
        logging.info(f"In training_step_end(): device={torch.cuda.current_device()}")
        if type(batch_parts) is torch.Tensor: 
            return batch_parts.mean()
          
        x, y, y_hat = batch_parts
        loss = self.criterion(x, y_hat, y)
        iou = iou_score(y_hat, y)
        dice = dice_coef(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)
        loss = self.criterion(x, y_hat, y)
        logging.debug(f"In validation_step(): device={torch.cuda.current_device()}, batch={batch_idx}, x.shape={x.shape}")
        return x, y, y_hat
  

    def validation_step_end(self, batch_parts):
        x, y, y_hat = batch_parts
        loss = self.criterion(x, y_hat, y)
        logging.debug(f"In validation_step_end(): device={torch.cuda.current_device()}, x.shape={x.shape}")
        return loss

    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)
        loss = self.criterion(x, y_hat, y)
        self.log('test_loss', loss, on_step=True)
        return loss

    def configure_optimizers(self):
        lr = 1e-3
        momentum = 0.9
        weight_decay = 1e-4
        params = filter(lambda p: p.requires_grad, self.parameters())
        sgd = optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        scheduler = lr_scheduler.CosineAnnealingLR(sgd, self.max_epochs)
        return [sgd], [scheduler]