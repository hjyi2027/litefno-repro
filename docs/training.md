# Training & Evaluation

## Run training

```bash
litefno train --config configs/experiments/litefno_gray_scott_reaction_diffusion.yaml
```

You can override config values on the CLI:

```bash
litefno train --config configs/experiments/litefno_gray_scott_reaction_diffusion.yaml --set training.epochs=10 --set training.device=cuda
```

Resume from a checkpoint:

```bash
litefno train --config configs/experiments/litefno_gray_scott_reaction_diffusion.yaml --set training.resume_from=/path/to/epoch_0100.pt
```

## What training does

- Loads processed splits from the dataset config (`train`, `valid`, `test` if present).
- Trains on the `train` split.
- Evaluates on the splits in `training.eval_splits` every `training.eval_every` epochs.
- Optionally evaluates on the `test` split at the end (`training.test_at_end`).
- Logs metrics to the JSONL path specified under `logging.metrics_path`.

## Metrics

The training loop reports:

- `loss` (MSE on the training batch)
- `rmse`, `vrmse`
- Optional windowed VRMSE metrics when `output_steps` covers the window (see `docs/metrics.md`).

All evaluation metrics are prefixed by split name, e.g. `train_vrmse`, `valid_vrmse_6_12`.

## Checkpoints

If configured, checkpoints are saved as:

```
training.checkpoint_dir/epoch_XXXX.pt
```

Set:

- `training.checkpoint_dir`
- `training.checkpoint_every`

## Reproducibility

- `training.seed`: sets Python/NumPy/PyTorch random seeds.
- `training.deterministic`: makes cuDNN deterministic (may reduce performance).

## Helpful config fields

See `docs/configs.md` for the full list. Common training fields:

- `epochs`, `batch_size`, `lr`, `lr_step`, `lr_gamma`
- `device` (`cpu` or `cuda`)
- `eval_every`, `eval_splits`, `eval_windows`
- `num_workers`, `pin_memory`
