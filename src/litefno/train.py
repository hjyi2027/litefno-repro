from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

from .config import apply_overrides, load_config
from .data import DatasetConfig, H5SequenceDataset
from .logging import MetricsLogger, setup_logging
from .metrics import rmse, vrmse
from .models import FNOS, LiteFNO


@dataclass
class TrainingConfig:
    epochs: int = 1
    batch_size: int = 4
    lr: float = 1e-3
    lr_step: int = 100
    lr_gamma: float = 0.5
    device: str = "cpu"


def infer_fields(path: Path, dataset_key: str) -> int:
    import h5py

    with h5py.File(path, "r") as handle:
        data = handle[dataset_key]
        return data.shape[-1]


def flatten_time(x: torch.Tensor) -> torch.Tensor:
    # (B, T, H, W, C) -> (B, T*C, H, W)
    if x.ndim != 5:
        raise ValueError("Expected 5D tensor (B, T, H, W, C).")
    b, t, h, w, c = x.shape
    return x.reshape(b, t * c, h, w)


def unflatten_time(x: torch.Tensor, time_steps: int, channels: int) -> torch.Tensor:
    # (B, T*C, H, W) -> (B, T, H, W, C)
    b, _, h, w = x.shape
    return x.reshape(b, time_steps, channels, h, w).permute(0, 1, 3, 4, 2)


def build_model(model_cfg: dict, in_channels: int, out_channels: int) -> nn.Module:
    name = model_cfg.get("name", "litefno").lower()
    width = model_cfg.get("width", 64)
    layers = model_cfg.get("layers", 8)
    if name == "fno_s":
        modes = model_cfg.get("modes", 12)
        return FNOS(in_channels, out_channels, width=width, modes=modes, layers=layers)
    rank = model_cfg.get("rank", 32)
    return LiteFNO(in_channels, out_channels, width=width, rank=rank, layers=layers)


def train_epoch(model: nn.Module, loader, optimizer, device: torch.device, output_steps: int, fields: int):
    model.train()
    total_loss = 0.0
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        inputs_flat = flatten_time(inputs)
        preds = model(inputs_flat)
        preds = unflatten_time(preds, output_steps, fields)
        loss = torch.nn.functional.mse_loss(preds, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(1, len(loader))


def evaluate(model: nn.Module, loader, device: torch.device, output_steps: int, fields: int):
    model.eval()
    total_rmse = 0.0
    total_vrmse = 0.0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            inputs_flat = flatten_time(inputs)
            preds = model(inputs_flat)
            preds = unflatten_time(preds, output_steps, fields)
            total_rmse += rmse(preds, targets).mean().item()
            total_vrmse += vrmse(preds, targets).mean().item()
    count = max(1, len(loader))
    return {"rmse": total_rmse / count, "vrmse": total_vrmse / count}


def run_training(config_path: Path, overrides=None) -> None:
    setup_logging()
    config = load_config(config_path)
    if overrides:
        config = apply_overrides(config, overrides)

    dataset_cfg = config["dataset"]
    dataset_key = dataset_cfg.get("dataset_key", "data")
    processed_dir = Path(dataset_cfg.get("processed_dir", dataset_cfg.get("output_dir", "")))
    data_dir = processed_dir / "train.h5"
    fields = dataset_cfg.get("fields") or infer_fields(data_dir, dataset_key)
    input_steps = dataset_cfg.get("input_steps", 1)
    output_steps = dataset_cfg.get("output_steps", 1)

    train_cfg = TrainingConfig(**config.get("training", {}))
    device = torch.device(train_cfg.device)

    ds_config = DatasetConfig(
        path=data_dir,
        dataset_key=dataset_key,
        input_steps=input_steps,
        output_steps=output_steps,
        stride=dataset_cfg.get("stride", 1),
        cache=dataset_cfg.get("cache", "none"),
    )
    dataset = H5SequenceDataset(ds_config)
    loader = torch.utils.data.DataLoader(dataset, batch_size=train_cfg.batch_size, shuffle=True)

    in_channels = input_steps * fields
    out_channels = output_steps * fields
    model = build_model(config.get("model", {}), in_channels, out_channels).to(device)
    optimizer = AdamW(model.parameters(), lr=train_cfg.lr)
    scheduler = StepLR(optimizer, step_size=train_cfg.lr_step, gamma=train_cfg.lr_gamma)

    logger = MetricsLogger(Path(config.get("logging", {}).get("metrics_path", "outputs/logs/metrics.jsonl")))

    for epoch in range(train_cfg.epochs):
        loss = train_epoch(model, loader, optimizer, device, output_steps, fields)
        metrics = evaluate(model, loader, device, output_steps, fields)
        logger.log(epoch, {"loss": loss, **metrics})
        scheduler.step()
