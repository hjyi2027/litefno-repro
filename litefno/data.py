from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class DatasetConfig:
    path: Path
    dataset_key: str = "data"
    input_steps: int = 1
    output_steps: int = 1
    stride: int = 1
    cache: str = "none"  # "none" or "memory"


class H5SequenceDataset(Dataset):
    def __init__(self, config: DatasetConfig):
        self.config = config
        self._file: h5py.File | None = None
        self._data: np.ndarray | None = None
        self._shape = None
        self._offsets = None
        self._load_shape()
        if config.cache == "memory":
            self._data = self._load_array()

    def _load_shape(self) -> None:
        with h5py.File(self.config.path, "r") as handle:
            if self.config.dataset_key not in handle:
                raise KeyError(f"Dataset key '{self.config.dataset_key}' not found in {self.config.path}")
            self._shape = handle[self.config.dataset_key].shape
        if len(self._shape) != 5:
            raise ValueError(f"Expected 5D array (n_traj, n_steps, H, W, fields); got {self._shape}")
        n_steps = self._shape[1]
        window_count = max(0, n_steps - self.config.input_steps - self.config.output_steps + 1)
        self._offsets = list(range(0, window_count, max(1, self.config.stride)))

    def _ensure_file(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.config.path, "r")
        return self._file

    def _load_array(self) -> np.ndarray:
        with h5py.File(self.config.path, "r") as handle:
            return handle[self.config.dataset_key][...]

    def __len__(self) -> int:
        if not self._offsets:
            return 0
        return self._shape[0] * len(self._offsets)

    def __getitem__(self, idx: int):
        if not self._offsets:
            raise IndexError("Dataset has no available windows.")
        offsets = self._offsets
        traj_idx = idx // len(offsets)
        t0 = offsets[idx % len(offsets)]
        t1 = t0 + self.config.input_steps
        t2 = t1 + self.config.output_steps
        if self._data is not None:
            source = self._data
        else:
            source = self._ensure_file()[self.config.dataset_key]
        data = source[traj_idx, t0:t2]
        inputs = data[: self.config.input_steps]
        targets = data[self.config.input_steps :]
        return torch.from_numpy(inputs.astype(np.float32)), torch.from_numpy(targets.astype(np.float32))

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


def build_dataloader(config: DatasetConfig, batch_size: int, shuffle: bool = True) -> DataLoader:
    dataset = H5SequenceDataset(config)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def iter_splits(base_dir: Path) -> Iterator[Path]:
    for split in ("train", "valid", "test"):
        candidate = base_dir / f"{split}.h5"
        if candidate.exists():
            yield candidate
