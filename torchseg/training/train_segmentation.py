import os

import cv2
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from torchseg.configuration.config import Config
from torchseg.configuration.state import State
from torchseg.data.drive_dataset import get_data_loaders
from torchseg.modeling.combine_net import CombineNet
from torchseg.modeling.dirnet_attention import DirNet
from torchseg.modeling.deeplab_torchvision import DeepLabV3
from torchseg.modeling.losses.dice import DiceBCELoss, DiceLoss
from torchseg.utils.metrics import MetricMaker
from torchseg.utils.visualization import visualization_binary, visualization_feature_maps


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
        self.model.load_state_dict(torch.load("checkpoint_best.pth"))

    def train(self):
        for epoch in range(self.config.training.num_epochs):
            self.train_epoch(epoch)
            self.validate_epoch(epoch)
            if State.test in self.data_loaders:
                self.test_epoch(epoch)
            if epoch % self.config.training.save_every == 0:
                torch.save(self.model.state_dict(), f"checkpoint.pth")

    def train_epoch(self, epoch):
        torch.cuda.empty_cache()
        train_bar = tqdm(self.data_loaders[State.train])
        self.model.train()
        self.scaler = torch.cuda.amp.GradScaler()
        self.metrics[State.train].reset()

        for idx, data in enumerate(train_bar):
            self.optimizer.zero_grad()
            image, mask = [d.to(self.config.device) for d in data]
            with torch.cuda.amp.autocast(enabled=True):
                outputs = self.model(image)
                loss = self.loss(outputs, mask)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            prediction = torch.argmax(outputs, dim=1)
            self.metrics[State.train].update(prediction, mask)
            if idx % 100 == 0:
                self.writer.add_images("train/images", image, epoch * len(self.data_loaders[State.train]) + idx)
                self.writer.add_scalar("train/loss", loss.item(), epoch * len(self.data_loaders[State.train]) + idx)
                self.writer.add_images("train/visualization", visualization_binary(mask, prediction), epoch * len(self.data_loaders[State.train]) + idx)
                self.writer.add_figure("train/feature_maps", visualization_feature_maps(outputs), epoch * len(self.data_loaders[State.train]) + idx)
            train_bar.set_description(f"Epoch: {epoch}, Loss: {loss.item():.4f}")
        for cls_, value in self.metrics[State.train].mean_iou.items():
            self.writer.add_scalar(f"TRAIN_MEAN_IOU/{cls_}", value, epoch)

        for cls_, value in self.metrics[State.train].mean_dice.items():
            self.writer.add_scalar(f"TRAIN_MEAN_DICE/{cls_}", value, epoch)

    @torch.no_grad()
    def validate_epoch(self, epoch):
        torch.cuda.empty_cache()
        validation_bar = tqdm(self.data_loaders[State.val])
        self.model.eval()
        self.metrics[State.val].reset()

        for idx, data in enumerate(validation_bar):
            image, mask = [d.to(self.config.device) for d in data]
            with torch.cuda.amp.autocast(enabled=False):
                outputs = self.model(image)
                loss = self.loss(outputs, mask)
            prediction = torch.argmax(outputs, dim=1)
            self.metrics[State.val].update(prediction, mask)
            if idx % 100 == 0:
                self.writer.add_images("val/images", image, epoch * len(self.data_loaders[State.val]) + idx)
                self.writer.add_images("val/visualization", visualization_binary(mask, prediction), epoch * len(self.data_loaders[State.val]) + idx)
                self.writer.add_figure("val/feature_maps", visualization_feature_maps(outputs), epoch * len(self.data_loaders[State.val]) + idx)
                self.writer.add_scalar("val/loss", loss.item(), epoch * len(self.data_loaders[State.val]) + idx)
            validation_bar.set_description(f"Epoch: {epoch}")
        for cls_, value in self.metrics[State.val].mean_iou.items():
            self.writer.add_scalar(f"VAL_MEAN_IOU/{cls_}", value, epoch)

        for cls_, value in self.metrics[State.val].mean_dice.items():
            self.writer.add_scalar(f"VAL_MEAN_DICE/{cls_}", value, epoch)

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
