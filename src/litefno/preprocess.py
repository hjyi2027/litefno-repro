from __future__ import annotations

from pathlib import Path
from typing import Optional

import h5py
import numpy as np

from .config import load_config, resolve_path


def downsample_spatial(array: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return array
    height = array.shape[2] - (array.shape[2] % factor)
    width = array.shape[3] - (array.shape[3] % factor)
    if height != array.shape[2] or width != array.shape[3]:
        array = array[:, :, :height, :width, :]
    reshaped = array.reshape(
        array.shape[0],
        array.shape[1],
        height // factor,
        factor,
        width // factor,
        factor,
        array.shape[4],
    )
    return reshaped.mean(axis=(3, 5))


def cap_trajectories(
    array: np.ndarray,
    max_trajectories: Optional[int],
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    if max_trajectories is None or max_trajectories >= array.shape[0]:
        return array
    if rng is None:
        rng = np.random.default_rng()
    indices = rng.choice(array.shape[0], size=max_trajectories, replace=False)
    indices = np.sort(indices)
    return array[indices]


def cap_timesteps(array: np.ndarray, max_steps: Optional[int]) -> np.ndarray:
    if max_steps is None:
        return array
    return array[:, : max_steps]


def preprocess_array(
    array: np.ndarray,
    factor: int,
    max_trajectories: Optional[int],
    max_steps: Optional[int],
    random_seed: Optional[int] = None,
) -> np.ndarray:
    rng = np.random.default_rng(random_seed) if random_seed is not None else None
    array = cap_trajectories(array, max_trajectories, rng=rng)
    array = cap_timesteps(array, max_steps)
    array = downsample_spatial(array, factor)
    return array.astype(np.float32)


def preprocess_h5_file(
    input_path: Path,
    output_path: Path,
    dataset_key: str,
    factor: int,
    max_trajectories: Optional[int],
    max_steps: Optional[int],
    random_seed: Optional[int] = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(input_path, "r") as handle:
        data = handle[dataset_key][...]
    processed = preprocess_array(data, factor, max_trajectories, max_steps, random_seed)
    with h5py.File(output_path, "w") as handle:
        handle.create_dataset(dataset_key, data=processed, compression="gzip")


def preprocess_from_config(config_path: Path) -> None:
    config = load_config(config_path)
    dataset = config["dataset"]
    dataset_key = dataset.get("dataset_key", "data")
    input_dir = resolve_path(dataset, "input_dir")
    output_dir = resolve_path(dataset, "output_dir")
    factor = dataset.get("downsample_factor", 1)
    max_trajectories = dataset.get("max_trajectories")
    max_steps = dataset.get("max_steps")
    random_seed = dataset.get("random_seed")

    for split in dataset.get("splits", ["train", "valid", "test"]):
        input_path = input_dir / f"{split}.h5"
        output_path = output_dir / f"{split}.h5"
        preprocess_h5_file(
            input_path,
            output_path,
            dataset_key,
            factor,
            max_trajectories,
            max_steps,
            random_seed=random_seed,
        )
