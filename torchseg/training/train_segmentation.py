import os

import cv2
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from torchseg.configuration.config import Config
from torchseg.configuration.state import State
from torchseg.data.drive_dataset import get_data_loaders
from torchseg.modeling.dirnet_attention import DirNet
from torchseg.modeling.losses.dice import DiceBCELoss
from torchseg.utils.metrics import MetricMaker


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
        self.loss = DiceBCELoss()
        self.metrics = {
            State.train: MetricMaker(["background", "vessel"]),
            State.val: MetricMaker(["background", "vessel"]),
        }

    def train(self):
        for epoch in range(self.config.training.num_epochs):
            self.train_epoch(epoch)
            self.validate_epoch(epoch)
            if State.test in self.data_loaders:
                self.test_epoch(epoch)
            if epoch % self.config.training.save_every == 0:
                torch.save(self.model.state_dict(), f"checkpoint.pth")

    def train_epoch(self, epoch):
        train_bar = tqdm(self.data_loaders[State.train])
        self.model.train()
        self.scaler = torch.cuda.amp.GradScaler()

        for idx, data in enumerate(train_bar):
            self.optimizer.zero_grad()
            image, mask = [d.to(self.config.device) for d in data]
            with torch.cuda.amp.autocast(enabled=True):
                outputs = self.model(image)
                loss = self.loss(outputs, mask)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            prediction = torch.argmax(outputs, dim=1).unsqueeze(1)
            self.metrics[State.train].update(prediction.squeeze(1), mask)
            if idx % 3 == 0:
                self.writer.add_images("train/prediction", prediction / 2., epoch * len(self.data_loaders[State.train]) + idx)
                self.writer.add_images("train/masks", mask.unsqueeze(1) / 2., epoch * len(self.data_loaders[State.train]) + idx)
                self.writer.add_images("train/images", image, epoch * len(self.data_loaders[State.train]) + idx)
                self.writer.add_scalar("train/loss", loss.item(), epoch * len(self.data_loaders[State.train]) + idx)
            train_bar.set_description(f"Epoch: {epoch}, Loss: {loss.item():.4f}")
        for cls_, value in self.metrics[State.train].mean_iou.items():
            self.writer.add_scalar(f"TRAIN_MEAN_IOU/{cls_}", value, epoch)

        for cls_, value in self.metrics[State.train].mean_dice.items():
            self.writer.add_scalar(f"TRAIN_MEAN_DICE/{cls_}", value, epoch)

    def validate_epoch(self, epoch):
        pass

    @torch.no_grad()
    def test_epoch(self, epoch):
        self.model.eval()

        for idx, data in enumerate(self.data_loaders[State.test]):
            image = data[0].to(self.config.device)
            output = self.model(image)
            prediction = torch.argmax(output, dim=1).unsqueeze(1)
            self.writer.add_images("test/prediction", prediction / 2., epoch * len(self.data_loaders[State.test]) + idx)
            self.writer.add_images("test/images", image, epoch * len(self.data_loaders[State.test]) + idx)


if __name__ == "__main__":
    config = Config()
    trainer = SegmentationTrainer(config)
    trainer.train()
