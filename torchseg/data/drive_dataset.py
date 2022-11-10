from email.mime import image
import glob
import os
from typing import List, Tuple

import albumentations as A
import cv2
import numpy as np
from PIL import Image
import torch
import torch.utils.data as data

from torchseg.configuration.state import State
from torchseg.utils.resize import resize, interp_methods


class SegmentationAnnotation:
    image_file: str
    mask_file: str


class DriveDataset(data.Dataset):

    def __init__(self, annotations: List[Tuple[str]], transforms: A.Compose, image_shape: int = 768) -> None:
        super().__init__()
        self.annotations = annotations
        self.transforms = transforms
        self.size = image_shape

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, index: int) -> dict:
        annotation = self.annotations[index]
        image = cv2.cvtColor(cv2.imread(annotation[0], cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB) / 255.
        if annotation[1]:
            mask = np.asarray(Image.open(annotation[1]))
        else:
            mask = np.zeros(image.shape[:2], dtype=np.float32) - 1
        image, mask = (resize(a, out_shape=(self.size, self.size), interp_method=interp_methods.linear).astype(np.float32) for a in [image.astype(np.float32), mask])
        mask[mask > 128] = 255
        transformed = self.transforms(image=image, mask=mask // 255)
        return (
            torch.from_numpy(transformed["image"].transpose(2, 0, 1)).float(),
            torch.from_numpy(transformed["mask"]).long()
        )


def get_data_loaders(config):
    annotations_train = list(zip(
        sorted(glob.glob(os.path.join(config.dataset.image_path, "*.tif"))),
        sorted(glob.glob(os.path.join(config.dataset.mask_path, "*.gif")))))

    annotations_val = list(zip(
        sorted(glob.glob(os.path.join(config.dataset.test_image_path, "*.tif"))),
        sorted(glob.glob(os.path.join(config.dataset.test_maks_path, "*.gif")))))

    train_dataset = DriveDataset(annotations_train, config.dataset.transforms[State.train], image_shape=config.image_size)
    val_dataset = DriveDataset(annotations_val, config.dataset.transforms[State.val], image_shape=config.image_size)
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    return {State.train: train_loader, State.val: val_loader}




def get_data_loaders_semi_supervised(config):
    annotations_train = list(zip(
        sorted(glob.glob(os.path.join(config.dataset.image_path, "*.tif"))),
        sorted(glob.glob(os.path.join(config.dataset.mask_path, "*.gif")))))

    annotations_val = list(zip(
        sorted(glob.glob(os.path.join(config.dataset.test_image_path, "*.tif"))),
        sorted(glob.glob(os.path.join(config.dataset.test_maks_path, "*.gif")))))

    annotations_unsupervised = [(ann[0], None) for ann in annotations_val]

    train_dataset = DriveDataset(annotations_train + annotations_unsupervised, config.dataset.transforms[State.train], image_shape=config.image_size)
    val_dataset = DriveDataset(annotations_val, config.dataset.transforms[State.val], image_shape=config.image_size)
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    return {State.train: train_loader, State.val: val_loader}
