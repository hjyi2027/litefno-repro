"""Preprocessing for The Well datasets.

Applies the paper's three reductions to raw data before training:
  1. Random trajectory sampling (up to ``max_trajectories``) with optional seed
  2. Timestep cap to ``max_steps``
  3. Anti-aliased spatial block-mean downsampling by ``downsample_factor``

Supports two input layouts:

  * **Well layout** (real downloads via ``litefno download``):
        ``<input_dir>/datasets/<dataset_name>/data/<split>/<regime>.hdf5``
    where each regime file has groups ``t0_fields/`` (scalars),
    ``t1_fields/`` (vectors), and ``t2_fields/`` (tensors). Fields are
    stacked along a trailing channel axis and regimes are concatenated
    along the trajectory axis.

  * **Flat layout** (tests, hand-crafted data):
        ``<input_dir>/<split>.h5`` with a single dataset at the top-level
    key (default ``data``) of shape ``(n_traj, n_steps, H, W, n_fields)``.

The output is always a single HDF5 per split in the flat layout, so the
training loader (``H5SequenceDataset``) does not need to change.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import h5py
import numpy as np

from .config import load_config, resolve_path


# ---------------------------------------------------------------------------
# Array-level reductions (unchanged from the original module)
# ---------------------------------------------------------------------------

def downsample_spatial(array: np.ndarray, factor: int) -> np.ndarray:
    """Anti-aliased spatial downsample via block-mean averaging.

    Operates on the H and W axes (positions 2 and 3) of a 5D array
    ``(n_traj, n_steps, H, W, n_fields)``. Trailing pixels that do not
    divide evenly are trimmed before averaging.
    """
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
    """Randomly sample up to ``max_trajectories`` along the trajectory axis."""
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
    return array[:, :max_steps]


def preprocess_array(
    array: np.ndarray,
    factor: int,
    max_trajectories: Optional[int],
    max_steps: Optional[int],
    random_seed: Optional[int] = None,
) -> np.ndarray:
    """Apply the three reductions to an already-5D array (flat-layout helper)."""
    rng = np.random.default_rng(random_seed) if random_seed is not None else None
    array = cap_trajectories(array, max_trajectories, rng=rng)
    array = cap_timesteps(array, max_steps)
    array = downsample_spatial(array, factor)
    return array.astype(np.float32)


# ---------------------------------------------------------------------------
# Flat layout (kept for backward compatibility with tests)
# ---------------------------------------------------------------------------

def preprocess_h5_file(
    input_path: Path,
    output_path: Path,
    dataset_key: str,
    factor: int,
    max_trajectories: Optional[int],
    max_steps: Optional[int],
    random_seed: Optional[int] = None,
) -> None:
    """Preprocess a single flat-layout HDF5 file with one top-level dataset."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(input_path, "r") as handle:
        data = handle[dataset_key][...]
    processed = preprocess_array(data, factor, max_trajectories, max_steps, random_seed)
    with h5py.File(output_path, "w") as handle:
        handle.create_dataset(dataset_key, data=processed, compression="gzip")


# ---------------------------------------------------------------------------
# Well layout (new)
# ---------------------------------------------------------------------------

def _load_well_file_as_5d(path: Path, max_steps: Optional[int] = None) -> np.ndarray:
    """Load one Well-format HDF5 file as a 5D ``(n_traj, n_steps, H, W, C)`` array.

    Iterates ``t0_fields/`` (scalar), ``t1_fields/`` (vector), and
    ``t2_fields/`` (tensor) groups in that order. Within each group, fields
    are sorted by name for deterministic channel ordering across regimes.

    Per-field reshape conventions:
      * scalar (4D ``(n_traj, n_steps, H, W)``) -> add trailing axis of size 1
      * vector (5D ``(n_traj, n_steps, H, W, d)``) -> kept as-is
      * tensor (6D ``(n_traj, n_steps, H, W, d, d)``) -> last two dims flattened

    Time-invariant fields (e.g. material parameters such as ``density`` in the
    acoustic-scattering regimes) drop The Well's time axis: a scalar arrives
    as 3D ``(n_traj, H, W)``, a vector as 4D, a tensor as 5D. These are
    broadcast along a synthetic time axis so they line up with the
    time-varying fields before channel concatenation.

    If ``max_steps`` is provided, the time axis is truncated *at read time*
    via h5py slicing, which avoids loading the full ~21 GB-per-regime file
    into memory for large Well datasets.
    """
    field_arrays: list[np.ndarray] = []
    expected_ndim = {"t0_fields": 4, "t1_fields": 5, "t2_fields": 6}
    with h5py.File(path, "r") as f:
        # Time axis size for this regime, used to broadcast time-invariant
        # fields. Prefer the explicit ``dimensions/time`` axis; otherwise peek
        # at the first time-varying field we find.
        if "dimensions/time" in f:
            full_time = int(f["dimensions/time"].shape[0])
        else:
            full_time = None
            for g in ("t0_fields", "t1_fields", "t2_fields"):
                if g not in f:
                    continue
                for name in f[g].keys():
                    obj = f[g][name]
                    if isinstance(obj, h5py.Dataset) and obj.ndim == expected_ndim[g]:
                        full_time = int(obj.shape[1])
                        break
                if full_time is not None:
                    break
        if max_steps is None:
            n_steps = full_time
        elif full_time is None:
            n_steps = max_steps
        else:
            n_steps = min(max_steps, full_time)

        for group_name in ("t0_fields", "t1_fields", "t2_fields"):
            if group_name not in f:
                continue
            for name in sorted(f[group_name].keys()):
                obj = f[group_name][name]
                if not isinstance(obj, h5py.Dataset):
                    continue
                if obj.ndim == expected_ndim[group_name] - 1:
                    if n_steps is None:
                        raise ValueError(
                            f"Cannot determine time-axis length to broadcast "
                            f"time-invariant field {group_name}/{name} in {path}"
                        )
                    raw = obj[...]
                    # broadcast_to returns a view; concatenate below materializes it.
                    arr = np.broadcast_to(
                        raw[:, np.newaxis], (raw.shape[0], n_steps, *raw.shape[1:])
                    )
                else:
                    arr = obj[:, :max_steps] if max_steps is not None else obj[...]
                if arr.ndim == 4:
                    arr = arr[..., np.newaxis]
                elif arr.ndim == 5:
                    pass
                elif arr.ndim == 6:
                    arr = arr.reshape(*arr.shape[:-2], arr.shape[-2] * arr.shape[-1])
                else:
                    raise ValueError(
                        f"Unexpected ndim={arr.ndim} for {group_name}/{name} in {path}"
                    )
                field_arrays.append(arr)
    if not field_arrays:
        raise ValueError(f"No t0/t1/t2_fields datasets found in {path}")
    return np.concatenate(field_arrays, axis=-1)


def preprocess_well_split(
    input_dir: Path,
    output_path: Path,
    dataset_name: str,
    split: str,
    dataset_key: str,
    factor: int,
    max_trajectories: Optional[int],
    max_steps: Optional[int],
    random_seed: Optional[int] = None,
) -> None:
    """Preprocess one split by walking The Well's nested directory layout.

    Memory strategy: read each regime file with timestep truncation already
    applied at the h5py level, then spatially downsample immediately. Only
    the small, reduced arrays accumulate in memory before concatenation.
    Trajectory sampling runs *after* concatenation so the sample is globally
    random across all regimes (paper-faithful).
    """
    split_dir = input_dir / "datasets" / dataset_name / "data" / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    files = sorted(split_dir.glob("*.hdf5")) + sorted(split_dir.glob("*.h5"))
    if not files:
        raise FileNotFoundError(f"No HDF5 files found under {split_dir}")

    regime_arrays: list[np.ndarray] = []
    for path in files:
        arr = _load_well_file_as_5d(path, max_steps=max_steps)
        arr = downsample_spatial(arr, factor)
        regime_arrays.append(arr)

    # Some Well datasets (e.g. viscoelastic_instability) have regimes with
    # different time-axis lengths. Truncate every regime to the shortest so
    # they can stack into one trajectory tensor.
    min_steps = min(arr.shape[1] for arr in regime_arrays)
    regime_arrays = [arr[:, :min_steps] for arr in regime_arrays]

    combined = np.concatenate(regime_arrays, axis=0)
    rng = np.random.default_rng(random_seed) if random_seed is not None else None
    combined = cap_trajectories(combined, max_trajectories, rng=rng)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as handle:
        handle.create_dataset(
            dataset_key, data=combined.astype(np.float32), compression="gzip"
        )


# ---------------------------------------------------------------------------
# Config entry point
# ---------------------------------------------------------------------------

def preprocess_from_config(config_path: Path) -> None:
    config = load_config(config_path)
    dataset = config["dataset"]
    dataset_name = dataset["name"]
    dataset_key = dataset.get("dataset_key", "data")
    input_dir = resolve_path(dataset, "input_dir")
    output_dir = resolve_path(dataset, "output_dir")
    factor = dataset.get("downsample_factor", 1)
    max_trajectories = dataset.get("max_trajectories")
    max_steps = dataset.get("max_steps")
    random_seed = dataset.get("random_seed")

    # Detect layout: real Well downloads land under
    # <input_dir>/datasets/<dataset_name>/data/<split>/<regime>.hdf5,
    # whereas tests and hand-crafted data use the flat
    # <input_dir>/<split>.h5 form.
    well_data_root = input_dir / "datasets" / dataset_name / "data"

    for split in dataset.get("splits", ["train", "valid", "test"]):
        output_path = output_dir / f"{split}.h5"
        if well_data_root.is_dir():
            preprocess_well_split(
                input_dir,
                output_path,
                dataset_name,
                split,
                dataset_key,
                factor,
                max_trajectories,
                max_steps,
                random_seed=random_seed,
            )
        else:
            input_path = input_dir / f"{split}.h5"
            preprocess_h5_file(
                input_path,
                output_path,
                dataset_key,
                factor,
                max_trajectories,
                max_steps,
                random_seed=random_seed,
            )
