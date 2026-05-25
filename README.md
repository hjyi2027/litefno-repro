# litefno-repro

Reproduction and extensions for the Lightweight Fourier Neural Operator (LITEFNO) paper, with an emphasis on low-resource deployment.

## Documentation

- [Project overview](docs/overview.md)
- [Setup](docs/setup.md)
- [Data & preprocessing](docs/data.md)
- [Training & evaluation](docs/training.md)
- [Reproduction guide](docs/reproduction.md)
- [Experiments](docs/experiments.md)
- [Configuration reference](docs/configs.md)
- [Metrics](docs/metrics.md)
- [Extensions roadmap](docs/extensions.md)

## Setup (quickstart)

```bash
conda create -n fno python=3.10
conda activate fno
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -e .
```

For development tools (tests):

```bash
pip install -e .[dev]
```

## Data (quickstart)

The project expects The Well datasets as HDF5 with shape `(n_traj, n_steps, H, W, fields)`.

Download (requires `the-well-download` to be installed and on PATH):

```bash
litefno download --config configs/datasets/gray_scott_reaction_diffusion.yaml
```

Preprocess (downsampling, trajectory/time caps):

```bash
litefno preprocess --config configs/datasets/gray_scott_reaction_diffusion.yaml
```

## Training & evaluation (quickstart)

```bash
litefno train --config configs/experiments/litefno_gray_scott_reaction_diffusion.yaml
```

Override config values on the CLI:

```bash
litefno train --config configs/experiments/litefno_gray_scott_reaction_diffusion.yaml --set training.epochs=10 --set training.device=cuda
```

Metrics are logged to the JSONL path in the config under `logging.metrics_path`.

## Tests

```bash
python -m pytest
```

GitHub Actions runs the same test command on pushes and pull requests via
`.github/workflows/tests.yml`.
