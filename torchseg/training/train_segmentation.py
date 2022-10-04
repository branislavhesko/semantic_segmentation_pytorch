import os

import cv2
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

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
            weight_decay=self.config.training.weight_decay
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
        self.loss = torch.nn.CrossEntropyLoss()

    def train(self):
        for epoch in range(self.config.training.num_epochs):
            self.train_epoch(epoch)
            self.validate_epoch(epoch)
            self.test_epoch(epoch)

    def train_epoch(self, epoch):
        train_bar = tqdm(self.data_loaders[State.train])
        self.model.train()
        self.scaler = torch.cuda.amp.GradScaler()

        for idx, data in enumerate(train_bar):
            self.optimizer.zero_grad()
            image, mask = [d.to(self.config.device) for d in data]
            with torch.cuda.amp.autocast():
                outputs = self.model(image)
                loss = self.loss(outputs, mask)
            prediction = torch.argmax(outputs, dim=1).unsqueeze(1)
            self.writer.add_images("train/prediction", prediction / 2., epoch * len(self.data_loaders[State.train]) + idx)
            self.writer.add_images("train/masks", mask.unsqueeze(1) / 2., epoch * len(self.data_loaders[State.train]) + idx)
            self.writer.add_images("train/images", image, epoch * len(self.data_loaders[State.train]) + idx)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.writer.add_scalar("train/loss", loss.item(), epoch * len(self.data_loaders[State.train]) + idx)
            train_bar.update()

    def validate_epoch(self, epoch):
        pass

    @torch.no_grad()
    def test_epoch(self, epoch):
        self.model.eval()

        for idx, data in enumerate(self.data_loaders[State.test]):
            image = data[0].to(self.config.device)
            with torch.cuda.amp.autocast():
                output = self.model(image)
            prediction = torch.argmax(output, dim=1).unsqueeze(1)
            self.writer.add_images("test/prediction", prediction / 2., epoch * len(self.data_loaders[State.test]) + idx)
            self.writer.add_images("test/images", image, epoch * len(self.data_loaders[State.test]) + idx)


if __name__ == "__main__":
    config = Config()
    trainer = SegmentationTrainer(config)
    trainer.train()
