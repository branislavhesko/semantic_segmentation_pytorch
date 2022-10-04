import dataclasses
import glob
import os
from typing import List

import albumentations as A
import cv2
import numpy as np
import torch
import torch.utils.data as data

from torchseg.configuration.state import State


class SegmentationAnnotation:
    image_file: str
    mask_file: str


class RavirDataset(data.Dataset):

    def __init__(self, annotations: List[SegmentationAnnotation], transforms: A.Compose) -> None:
        super().__init__()
        self.annotations = annotations
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, index: int) -> dict:
        annotation = self.annotations[index]
        image = cv2.cvtColor(cv2.imread(annotation[0], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        if annotation[1]:
            mask = cv2.imread(annotation[1], cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
        image, mask = (cv2.resize(a, (768, 768), interpolation=cv2.INTER_NEAREST) for a in [image, mask])
        transformed = self.transforms(image=image, mask=mask // 127)
        return (
            torch.from_numpy(transformed["image"].transpose(2, 0, 1)).float(),
            torch.from_numpy(transformed["mask"]).long()
        )


def get_data_loaders(config):
    annotations = list(zip(
        sorted(glob.glob(os.path.join(config.dataset.image_path, "*.png"))),
        sorted(glob.glob(os.path.join(config.dataset.mask_path, "*.png")))))

    if config.dataset.test_image_path:
        test_annotations = [
            (image, "") for image in sorted(glob.glob(os.path.join(config.dataset.test_image_path, "*.png")))
        ]
    transforms = config.dataset.transforms

    return {
        State.train: data.DataLoader(
            RavirDataset(annotations, transforms[State.train]),
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.training.num_workers
        ),
        State.val: data.DataLoader(
            RavirDataset(annotations, transforms[State.val]),
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.training.num_workers
        ),
        State.test: data.DataLoader(
            RavirDataset(test_annotations, transforms[State.val]),
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.training.num_workers
        )

    }
