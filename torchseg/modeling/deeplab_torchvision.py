import torch
from torchvision.models.segmentation.deeplabv3 import deeplabv3_resnet101


class DeepLabV3(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = deeplabv3_resnet101(pretrained=False, progress=True, num_classes=num_classes, aux_loss=None)

    def forward(self, x):
        return self.model(x)['out']
