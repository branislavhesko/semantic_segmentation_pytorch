import torch
from torch import nn
import torch.nn.functional as F


# PyTorch
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        BCE = F.cross_entropy(inputs, targets, weight=torch.tensor([1., 2.], device=targets.device), reduction='mean')

        inputs = F.sigmoid(inputs)
        # flatten label and prediction tensors
        labels_layered = torch.zeros_like(inputs)
        for idx in range(inputs.shape[1]):
            labels_layered[:, idx, :, :][targets == idx] = 1
        targets = labels_layered.view(-1)
        inputs = inputs.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        dice_bce = BCE + dice_loss
        return dice_bce


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        labels_layered = torch.zeros_like(inputs)
        for idx in range(inputs.shape[1]):
            labels_layered[:, idx, :, :][targets == idx] = 1
        targets = labels_layered.view(-1)
        inputs = inputs.view(-1)


        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice