from typing import Any, List, Tuple
import numpy as np
import torch.utils.data as data


class MergedDataset(data.Dataset):
    def __init__(self, datasets: List[data.Dataset]) -> None:
        super().__init__()
        self.datasets = datasets
        self.cum_lengths = np.cumsum([len(dataset) for dataset in datasets])

    def __len__(self) -> int:
        return sum(len(dataset) for dataset in self.datasets)

    def __getitem__(self, index: int):
        dataset_idx = np.searchsorted(self.cum_lengths, index, side="right")
        value_idx = index - self.cum_lengths[dataset_idx - 1] if dataset_idx > 0 else index
        return self.datasets[dataset_idx][value_idx]
