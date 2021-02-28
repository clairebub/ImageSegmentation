import torch
import torch.nn as nn
# import torch.nn.functional as F

try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def dice_coefficient(self, output, target):
        smooth = 1e-5
        output = torch.sigmoid(output)
        num = target.size(0)
        output = output.view(num, -1)
        target = target.view(num, -1)
        intersection = (output * target)
        dice = (2. * intersection.sum(1) + smooth) / (output.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num

        return dice

    def forward(self, input, output, target):

        dice = self.dice_coefficient(output, target)

        return dice


class BCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def bce(self, output, target):
        criterion = nn.BCEWithLogitsLoss()
        bce = criterion(output, target)

        return bce

    def forward(self, input, output, target):

        bce = self.bce(output, target)

        return bce


class WeightedBCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def weighted_bce(self, output, target):
        criterion = nn.BCEWithLogitsLoss(reduction='none')

        unweighted_bce = torch.mean(criterion(output, target))

        disease_loss = torch.mean(target * criterion(output, target))
        ratio = torch.sum(1 - target) / torch.sum(target)
        weighted_bce = unweighted_bce + (ratio - 1) * disease_loss

        return weighted_bce

    def forward(self, input, output, target):

        weighted_bce = self.weighted_bce(output, target)

        return weighted_bce


class BCEDiceLoss(DiceLoss, BCELoss):
    def __init__(self):
        super().__init__()

    def forward(self, input, output, target):

        dice = self.dice_coefficient(output, target)
        bce = self.bce(output, target)

        return 0.5 * bce + dice


class WeightedBCEDiceLoss(DiceLoss, WeightedBCELoss):
    def __init__(self):
        super().__init__()

    def forward(self, input, output, target):

        dice = self.dice_coefficient(output, target)
        weighted_bce = self.weighted_bce(output, target)

        return 0.5 * weighted_bce + dice


class BCEDiceLungLoss(BCEDiceLoss):
    def __init__(self):
        super().__init__()

    def lung_loss(self, output, target):
        output = torch.sigmoid(output)

        disease_output = output[:, 0, :, :]
        disease_output = disease_output > 0.5

        lung_target = target[:, 1, :, :]
        lung_loss_tensor = disease_output.float() - lung_target.float()
        lung_loss_tensor[lung_loss_tensor < 0] = 0

        lung_regularizer = torch.mean(lung_loss_tensor)

        return lung_regularizer

    def forward(self, input, output, target, sub_dataset='disease'):

        if sub_dataset == 'disease':
            dice = self.dice_coefficient(output[:, 0, :, :], target[:, 0, :, :])
            bce = self.bce(output[:, 0, :, :], target[:, 0, :, :])
        else:
            dice = self.dice_coefficient(output, target)
            bce = self.bce(output, target)

        lung_regularizer = self.lung_loss(output, target)

        return 0.5 * bce + dice + 0.2 * lung_regularizer


class WeightedBCEDiceLungLoss(WeightedBCEDiceLoss, BCEDiceLungLoss):
    def __init__(self):
        super().__init__()

    def forward(self, input, output, target, sub_dataset='disease'):

        if sub_dataset == 'disease':
            dice = self.dice_coefficient(output[:, 0, :, :], target[:, 0, :, :])
            weighted_bce = self.weighted_bce(output[:, 0, :, :], target[:, 0, :, :])
        else:
            dice = self.dice_coefficient(output, target)
            weighted_bce = self.weighted_bce(output, target)

        lung_regularizer = self.lung_loss(output, target)

        return 0.5 * weighted_bce + dice + 0.2 * lung_regularizer


class BCEDiceColorLoss(BCEDiceLoss):
    def __init__(self):
        super().__init__()

    def color_loss(self, input, output, target):
        criterion = nn.BCEWithLogitsLoss(reduction='none')

        color_loss_tensor = target * input * criterion(output, target)
        color_regularizer = torch.mean(color_loss_tensor)

        return color_regularizer

    def forward(self, input, output, target):

        dice = self.dice_coefficient(output, target)
        bce = self.bce(output, target)
        color_regularizer = self.color_loss(input, output, target)

        return 0.5 * bce + dice + 0.2 * color_regularizer


class WeightedBCEDiceColorLoss(WeightedBCEDiceLoss, BCEDiceColorLoss):
    def __init__(self):
        super().__init__()

    def forward(self, input, output, target):

        dice = self.dice_coefficient(output, target)
        weighted_bce = self.weighted_bce(output, target)
        color_regularizer = self.color_loss(input, output, target)

        return 0.5 * weighted_bce + dice + 0.2 * color_regularizer


# class BCEDiceLoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, input, output, target):
#         bce = F.binary_cross_entropy_with_logits(output, target)
#
#         smooth = 1e-5
#         output = torch.sigmoid(output)
#         num = target.size(0)
#         output = output.view(num, -1)
#         target = target.view(num, -1)
#         intersection = (output * target)
#         dice = (2. * intersection.sum(1) + smooth) / (output.sum(1) + target.sum(1) + smooth)
#         dice = 1 - dice.sum() / num
#
#         return 0.5 * bce + dice


# class LovaszHingeLoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, input, target):
#         input = input.squeeze(1)
#         target = target.squeeze(1)
#         loss = lovasz_hinge(input, target, per_image=True)
#
#         return loss
