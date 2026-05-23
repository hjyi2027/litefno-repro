# Experiments

This page collects the typical experiment workflows used during reproduction and extensions.

## Config-driven runs

Each experiment YAML specifies the dataset, model, and training parameters.

Example LITEFNO run:

```bash
litefno train --config configs/experiments/litefno_gray_scott_reaction_diffusion.yaml
```

## Sweeps

Sweep rank and width with CLI overrides:

```bash
litefno train --config configs/experiments/litefno_gray_scott_reaction_diffusion.yaml \
  --set model.width=128 \
  --set model.rank=16
```

Sweep the FNO-S baseline:

```bash
litefno train --config configs/experiments/fno_s_gray_scott_reaction_diffusion.yaml \
  --set model.width=128 \
  --set model.modes=16
```

## Checkpointing and resume

Enable checkpoints:

```bash
litefno train --config configs/experiments/litefno_gray_scott_reaction_diffusion.yaml \
  --set training.checkpoint_dir=outputs/checkpoints/gs_litefno \
  --set training.checkpoint_every=50
```

Resume from a checkpoint:

```bash
litefno train --config configs/experiments/litefno_gray_scott_reaction_diffusion.yaml \
  --set training.resume_from=outputs/checkpoints/gs_litefno/epoch_0050.pt
```

## Logging

Metrics are written to the JSONL file under `logging.metrics_path`. Each line is a JSON object for a given epoch, making it easy to load with pandas for plotting.
