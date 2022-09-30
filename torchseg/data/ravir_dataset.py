import dataclasses
import glob
import os
from typing import List

import albumentations as A
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
        image = A.io.read(annotation.image_file)
        mask = A.io.read(annotation.mask_file)
        return self.transforms(image=image, mask=mask)


def get_data_loaders(config):
    annotations = list(zip(
        sorted(glob.glob(os.path.join(config.dataset.image_path, "*.png"))),
        sorted(glob.glob(os.path.join(config.dataset.mask_path, "*.png")))))
    transforms = config.dataset.transforms

    return {
        State.train: data.DataLoader(
            RavirDataset(annotations, transforms),
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.training.num_workers
        ),
        State.val: data.DataLoader(
            RavirDataset(annotations, transforms),
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.training.num_workers
        )
    }
