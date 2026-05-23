# Data & Preprocessing

## Dataset source

All datasets are from The Well benchmark (Polymathic AI). Do **not** download the full collection (≈15TB). Start with the smallest dataset and only pull the splits you need.

## Expected format

The raw data are HDF5 arrays with shape:

```
(n_traj, n_steps, H, W, fields)
```

The pipeline down-samples spatially, caps the number of trajectories, and caps the time dimension to match the paper’s preprocessing.

## Download

Download a dataset split using the YAML config:

```bash
litefno download --config configs/datasets/gray_scott_reaction_diffusion.yaml
```

Repeat for other datasets by swapping the config file.

## Preprocess

```bash
litefno preprocess --config configs/datasets/gray_scott_reaction_diffusion.yaml
```

Preprocessing applies:

- `downsample_factor`: spatial stride (e.g., 4×).
- `max_trajectories`: cap on the number of trajectories (paper: ≤1000).
- `max_steps`: cap on time steps (paper: 30–60, dataset-dependent).

## Dataset config fields

Each dataset config lives under `configs/datasets/` and uses:

- `name`: The Well dataset name.
- `input_dir`: directory for raw splits.
- `output_dir`: directory for processed splits.
- `splits`: list of splits to download/preprocess (`train`, `valid`, `test`).
- `dataset_key`: HDF5 dataset key (default `data`).
- `downsample_factor`: spatial stride factor.
- `max_trajectories`: maximum trajectories to keep.
- `max_steps`: maximum time steps to keep.
- `input_steps`: number of input frames.
- `output_steps`: number of target frames.
- `fields`: number of physical fields (channels).
- `stride`: temporal stride between windows.
- `cache`: `none` or `memory`.
