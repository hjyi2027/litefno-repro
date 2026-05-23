# Setup

## Environment

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

## External tools

The data download step requires the `the-well-download` CLI to be available on your PATH. Install it according to The Well dataset instructions, then verify:

```bash
the-well-download --help
```

## Quick sanity check

```bash
litefno --help
```
