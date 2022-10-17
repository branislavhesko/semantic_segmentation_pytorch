import dataclasses
from typing import Dict

import albumentations as A
import cv2
import torch

from torchseg.configuration.state import State


def get_transforms():
    return {
        State.train: A.Compose([
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.0, always_apply=False, p=0.5),
            A.Rotate(limit=15, interpolation=cv2.INTER_LINEAR, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(),
            A.RandomSizedCrop(min_max_height=(384, 768), height=768, width=768, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2),
        ]),
        State.val: A.Compose([])
    }



@dataclasses.dataclass()
class TrainingConfig:
    num_epochs: int = 300
    batch_size: int = 2
    num_workers: int = 4
    lr: float = 2e-4
    weight_decay: float = 1e-4
    amp_enabled: bool = True
    save_every = 20


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
class DriveDatasetConfig:
    transforms: Dict[State, A.Compose] = dataclasses.field(default_factory=get_transforms)
    image_path: str = "/home/brani/doktorat/semantic_segmentation_pytorch/data/DRIVE/training/images"
    mask_path: str = "/home/brani/doktorat/semantic_segmentation_pytorch/data/DRIVE/training/1st_manual"
    test_image_path: str = "/home/brani/doktorat/semantic_segmentation_pytorch/data/DRIVE/test/images"
    test_maks_path: str = "/home/brani/doktorat/semantic_segmentation_pytorch/data/DRIVE/test/1st_manual"


@dataclasses.dataclass()
class Config:
    training: TrainingConfig = TrainingConfig()
    dataset: RavirDatasetConfig | DriveDatasetConfig = DriveDatasetConfig()
    model: DirNetConfig = DirNetConfig()
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes: int = 2
    image_size: int = 768