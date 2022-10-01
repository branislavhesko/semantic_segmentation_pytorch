import os

import cv2
import torch
from torch.utils.tensorboard import SummaryWriter

from torchseg.configuration.config import Config
from torchseg.configuration.state import State
from torchseg.data.ravir_dataset import get_data_loaders
from torchseg.modeling.dirnet import DirNet


class SegmentationTrainer:

    def __init__(self, config: Config) -> None:
        self.config = config
        self.model = DirNet(
            num_classes=self.config.num_classes
        ).to(self.config.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.training.lr,
            weight_decay=self.config.weight_decay
        )
        # TODO: use this.
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.1,
            patience=5,
        )
        self.writer = SummaryWriter()
        self.data_loaders = get_data_loaders(self.config)

    def train(self):
        for epoch in range(self.config.training.num_epochs):
            self.train_epoch(epoch)
            self.validate_epoch(epoch)
            self.test_epoch(epoch)

    def train_epoch(self, epoch):
        pass

    def validate_epoch(self, epoch):
        pass

    def test_epoch(self, epoch):
        pass
