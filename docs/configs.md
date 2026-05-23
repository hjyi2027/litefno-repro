# Configuration reference

## Config structure

Experiment configs live in `configs/experiments/` and often use a `base_config` field to inherit defaults:

```yaml
base_config: base_litefno.yaml
dataset:
  name: gray_scott_reaction_diffusion
  processed_dir: data/processed/gray_scott_reaction_diffusion
```

The loader merges base configs with dataset-specific overrides.

## Dataset config (training)

Used by `litefno train`:

- `processed_dir`: directory containing processed `train.h5`, `valid.h5`, `test.h5`.
- `dataset_key`: dataset key inside the HDF5 file (default `data`).
- `input_steps`, `output_steps`: temporal window sizes.
- `fields`: number of physical channels.
- `stride`: temporal stride between windows.
- `cache`: `none` or `memory`.
- `splits`: list of splits to load (default `train`, `valid`, `test`).

## Dataset config (download/preprocess)

Used by `litefno download` and `litefno preprocess`:

- `name`: The Well dataset name.
- `input_dir`: raw dataset folder.
- `output_dir`: processed dataset folder.
- `downsample_factor`: spatial downsampling factor.
- `max_trajectories`: cap on number of trajectories.
- `max_steps`: cap on time steps.
- `splits`: which splits to operate on.

## Model config

- `name`: `litefno` or `fno_s`.
- `layers`: number of blocks.
- `width`: hidden width.
- `rank` (LiteFNO only): low-rank bottleneck size.
- `modes` (FNO-S only): number of Fourier modes.

## Training config

- `epochs`, `batch_size`, `lr`, `lr_step`, `lr_gamma`, `device`
- `eval_every`: evaluate every N epochs (0 disables).
- `eval_splits`: list of splits to evaluate (e.g., `["train", "valid"]`).
- `eval_windows`: list of `[start, end]` time windows for VRMSE.
- `test_at_end`: run test metrics after training.
- `checkpoint_dir`, `checkpoint_every`: checkpoint output configuration.
- `num_workers`, `pin_memory`: DataLoader settings.

## Logging config

- `metrics_path`: JSONL output file for metrics.
