# Project Overview

This repository reproduces the Lightweight Fourier Neural Operator (LITEFNO) for time-dependent PDEs and adds extensions for low-resource deployment and social empowerment. The codebase is intentionally config-driven and lightweight, so experiments remain reproducible even on constrained hardware.

## Goals

- Reproduce LITEFNO and FNO-S baselines on The Well datasets.
- Document every experimental detail: preprocessing, hyperparameters, model sizes, and metrics.
- Extend the work toward low-resource settings (smaller ranks, quantization, robustness, explainability).

## Repository layout

- `configs/`: dataset and experiment configurations (base configs + per-dataset overrides).
- `src/litefno/`: CLI, data pipeline, preprocessing, models, training, and metrics.
- `docs/`: documentation pages for setup, data, training, metrics, and extensions.
- `scripts/`: thin wrappers around the CLI (optional).
- `tests/`: unit tests for preprocessing and metrics.

## Phases (roadmap)

1. **Setup**: environment and dependencies.
2. **Data preparation**: download + preprocess The Well datasets.
3. **Reproduction**: LITEFNO + FNO-S baselines with paper hyperparameters.
4. **Extensions**: low-rank sweeps, quantization, robustness, and explainability.
5. **Writing**: compile results and insights into a reproducibility report.
