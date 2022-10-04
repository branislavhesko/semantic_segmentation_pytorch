import dataclasses
from typing import Dict

import albumentations as A
import torch

from torchseg.configuration.state import State


def get_transforms():
    return {
        State.train: A.Compose([
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, always_apply=False, p=0.5),
            A.Rotate(limit=15, interpolation=1, border_mode=4, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate()
        ]),
        State.val: A.Compose([])
    }



@dataclasses.dataclass()
class TrainingConfig:
    num_epochs: int = 100
    batch_size: int = 2
    num_workers: int = 4
    lr: float = 2e-4
    weight_decay: float = 1e-4
    amp_enabled: bool = True


# TODO: finish
@dataclasses.dataclass()
class DirNetConfig:
    angles: torch.Tensor = torch.tensor([0, 30, 45, 60, 90, 120, 135, 150]).float()
    in_channels: int = 64
    out_channels: int = 384


@dataclasses.dataclass()
class RavirDatasetConfig:
    transforms: Dict[State, A.Compose] = dataclasses.field(default_factory=get_transforms)
    image_path: str = "/home/brani/doktorat/semantic_segmentation_pytorch/data/RAVIR Dataset/train/training_images"
    mask_path: str = "/home/brani/doktorat/semantic_segmentation_pytorch/data/RAVIR Dataset/train/training_masks"
    test_image_path: str = "/home/brani/doktorat/semantic_segmentation_pytorch/data/RAVIR Dataset/test/"

@dataclasses.dataclass()
class Config:
    training: TrainingConfig = TrainingConfig()
    dataset: RavirDatasetConfig = RavirDatasetConfig()
    model: DirNetConfig = DirNetConfig()
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes: int = 3