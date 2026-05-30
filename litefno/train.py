from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import random

import numpy as np

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

from tqdm.auto import tqdm

from .config import apply_overrides, load_config
from .data import DatasetConfig, H5SequenceDataset
from .logging import MetricsLogger, setup_logging
from .metrics import rmse, vrmse, windowed_vrmse
from .models import FNOS, LiteFNO


@dataclass
class TrainingConfig:
    epochs: int = 1
    batch_size: int = 4
    lr: float = 1e-3
    lr_step: int = 100
    lr_gamma: float = 0.5
    device: str = "cpu"
    seed: int | None = 1337
    deterministic: bool = False
    eval_every: int = 1
    eval_splits: Sequence[str] = ("train", "valid")
    eval_windows: Sequence[Sequence[int]] = ((6, 12), (13, 30))
    test_at_end: bool = True
    checkpoint_dir: str | None = None
    checkpoint_every: int = 0
    checkpoint_best_metric: str | None = None
    resume_from: str | None = None
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    prefetch_factor: int | None = None
    amp: bool = False
    cudnn_benchmark: bool = False


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
    x = x.permute(0, 1, 4, 2, 3)
    return x.reshape(b, t * c, h, w)


def unflatten_time(x: torch.Tensor, time_steps: int, channels: int) -> torch.Tensor:
    # (B, T*C, H, W) -> (B, T, H, W, C)
    b, _, h, w = x.shape
    x = x.reshape(b, time_steps, channels, h, w)
    return x.permute(0, 1, 3, 4, 2)


def build_model(model_cfg: dict, in_channels: int, out_channels: int) -> nn.Module:
    name = model_cfg.get("name", "litefno").lower()
    width = model_cfg.get("width", 64)
    layers = model_cfg.get("layers", 8)
    if name == "fno_s":
        modes = model_cfg.get("modes", 12)
        return FNOS(in_channels, out_channels, width=width, modes=modes, layers=layers)
    rank = model_cfg.get("rank", 32)
    return LiteFNO(in_channels, out_channels, width=width, rank=rank, layers=layers)


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def set_seed(seed: int, deterministic: bool = False, cudnn_benchmark: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif cudnn_benchmark:
        torch.backends.cudnn.benchmark = True


def normalize_windows(windows: Sequence[Sequence[int]], output_steps: int) -> list[tuple[int, int]]:
    valid = []
    for window in windows:
        if len(window) != 2:
            raise ValueError(f"Window should be (start, end): {window}")
        start, end = int(window[0]), int(window[1])
        if end <= start:
            raise ValueError(f"Window end must be greater than start: {window}")
        if output_steps >= end:
            valid.append((start, end))
    return valid


def build_dataloaders(
    dataset_cfg: dict,
    batch_size: int,
    dataset_key: str,
    input_steps: int,
    output_steps: int,
    stride: int,
    cache: str,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool = False,
    prefetch_factor: int | None = None,
) -> dict[str, torch.utils.data.DataLoader]:
    processed_dir = Path(dataset_cfg.get("processed_dir", dataset_cfg.get("output_dir", "")))
    splits = dataset_cfg.get("splits", ["train", "valid", "test"])
    loaders: dict[str, torch.utils.data.DataLoader] = {}
    for split in splits:
        path = processed_dir / f"{split}.h5"
        if not path.exists():
            continue
        ds_config = DatasetConfig(
            path=path,
            dataset_key=dataset_key,
            input_steps=input_steps,
            output_steps=output_steps,
            stride=stride,
            cache=cache,
        )
        dataset = H5SequenceDataset(ds_config)
        loader_kwargs = dict(
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = persistent_workers
            if prefetch_factor is not None:
                loader_kwargs["prefetch_factor"] = prefetch_factor
        loader = torch.utils.data.DataLoader(dataset, **loader_kwargs)
        loaders[split] = loader
    if "train" not in loaders:
        raise FileNotFoundError(f"Training split not found under {processed_dir}")
    return loaders


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    path: Path,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    device: torch.device | None = None,
) -> int:
    state = torch.load(path, map_location=device or "cpu")
    model.load_state_dict(state["model_state"])
    if optimizer is not None and "optimizer_state" in state:
        optimizer.load_state_dict(state["optimizer_state"])
    if scheduler is not None and "scheduler_state" in state:
        scheduler.load_state_dict(state["scheduler_state"])
    return int(state.get("epoch", 0))


def train_epoch(
    model: nn.Module,
    loader,
    optimizer,
    device: torch.device,
    output_steps: int,
    fields: int,
    scaler: "torch.amp.GradScaler | None" = None,
    amp: bool = False,
):
    model.train()
    total_loss = 0.0
    non_blocking = device.type == "cuda"
    use_amp = amp and device.type == "cuda"
    pbar = tqdm(loader, desc="train", leave=False, dynamic_ncols=True)
    for step, (inputs, targets) in enumerate(pbar, start=1):
        inputs = inputs.to(device, non_blocking=non_blocking)
        targets = targets.to(device, non_blocking=non_blocking)
        inputs_flat = flatten_time(inputs)
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.amp.autocast("cuda"):
                preds = model(inputs_flat)
                preds = unflatten_time(preds, output_steps, fields)
                loss = torch.nn.functional.mse_loss(preds, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            preds = model(inputs_flat)
            preds = unflatten_time(preds, output_steps, fields)
            loss = torch.nn.functional.mse_loss(preds, targets)
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{total_loss / step:.4e}")
    pbar.close()
    return total_loss / max(1, len(loader))


def evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
    output_steps: int,
    fields: int,
    windows: Sequence[tuple[int, int]] | None = None,
    amp: bool = False,
):
    model.eval()
    total_rmse = 0.0
    total_vrmse = 0.0
    window_sums = {}
    non_blocking = device.type == "cuda"
    use_amp = amp and device.type == "cuda"
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="eval", leave=False, dynamic_ncols=True):
            inputs = inputs.to(device, non_blocking=non_blocking)
            targets = targets.to(device, non_blocking=non_blocking)
            inputs_flat = flatten_time(inputs)
            if use_amp:
                with torch.amp.autocast("cuda"):
                    preds = model(inputs_flat)
                preds = preds.float()
            else:
                preds = model(inputs_flat)
            preds = unflatten_time(preds, output_steps, fields)
            total_rmse += rmse(preds, targets).mean().item()
            total_vrmse += vrmse(preds, targets).mean().item()
            if windows:
                window_metrics = windowed_vrmse(preds, targets, windows=windows)
                for key, value in window_metrics.items():
                    window_sums[key] = window_sums.get(key, 0.0) + value.mean().item()
    count = max(1, len(loader))
    metrics = {"rmse": total_rmse / count, "vrmse": total_vrmse / count}
    for key, value in window_sums.items():
        metrics[key] = value / count
    return metrics


def run_evaluation(
    config_path: Path,
    checkpoint: Path,
    split: str = "test",
    overrides=None,
) -> dict:
    setup_logging()
    config = load_config(config_path)
    if overrides:
        config = apply_overrides(config, overrides)

    dataset_cfg = config["dataset"]
    dataset_key = dataset_cfg.get("dataset_key", "data")
    processed_dir = Path(dataset_cfg.get("processed_dir", dataset_cfg.get("output_dir", "")))
    train_path = processed_dir / "train.h5"
    fields = dataset_cfg.get("fields") or infer_fields(train_path, dataset_key)
    input_steps = dataset_cfg.get("input_steps", 1)
    output_steps = dataset_cfg.get("output_steps", 1)

    train_cfg = TrainingConfig(**config.get("training", {}))
    device = torch.device(train_cfg.device if torch.cuda.is_available() or train_cfg.device == "cpu" else "cpu")
    eval_windows = normalize_windows(train_cfg.eval_windows, output_steps) if train_cfg.eval_windows else []

    loaders = build_dataloaders(
        dataset_cfg=dataset_cfg,
        batch_size=train_cfg.batch_size,
        dataset_key=dataset_key,
        input_steps=input_steps,
        output_steps=output_steps,
        stride=dataset_cfg.get("stride", 1),
        cache=dataset_cfg.get("cache", "none"),
        num_workers=train_cfg.num_workers,
        pin_memory=train_cfg.pin_memory,
        persistent_workers=train_cfg.persistent_workers,
        prefetch_factor=train_cfg.prefetch_factor,
    )
    if split not in loaders:
        raise FileNotFoundError(f"Split '{split}' not found under {processed_dir}")

    in_channels = input_steps * fields
    out_channels = output_steps * fields
    model = build_model(config.get("model", {}), in_channels, out_channels).to(device)
    load_checkpoint(Path(checkpoint), model, device=device)
    use_amp = train_cfg.amp and device.type == "cuda"

    metrics = evaluate(
        model,
        loaders[split],
        device,
        output_steps,
        fields,
        windows=eval_windows,
        amp=use_amp,
    )

    logger = MetricsLogger(Path(config.get("logging", {}).get("metrics_path", "outputs/logs/metrics.jsonl")))
    logger.log(-1, {f"{split}_{k}": v for k, v in metrics.items()})

    print(f"[{split}] checkpoint={checkpoint}")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6e}")
    return metrics


def run_training(config_path: Path, overrides=None) -> None:
    setup_logging()
    config = load_config(config_path)
    if overrides:
        config = apply_overrides(config, overrides)

    dataset_cfg = config["dataset"]
    dataset_key = dataset_cfg.get("dataset_key", "data")
    processed_dir = Path(dataset_cfg.get("processed_dir", dataset_cfg.get("output_dir", "")))
    train_path = processed_dir / "train.h5"
    fields = dataset_cfg.get("fields") or infer_fields(train_path, dataset_key)
    input_steps = dataset_cfg.get("input_steps", 1)
    output_steps = dataset_cfg.get("output_steps", 1)

    train_cfg = TrainingConfig(**config.get("training", {}))
    device = torch.device(train_cfg.device)
    eval_windows = normalize_windows(train_cfg.eval_windows, output_steps) if train_cfg.eval_windows else []
    if isinstance(train_cfg.eval_splits, str):
        eval_splits = [train_cfg.eval_splits]
    else:
        eval_splits = list(train_cfg.eval_splits)
    if train_cfg.seed is not None:
        set_seed(train_cfg.seed, train_cfg.deterministic, train_cfg.cudnn_benchmark)
    elif train_cfg.cudnn_benchmark and not train_cfg.deterministic:
        torch.backends.cudnn.benchmark = True

    loaders = build_dataloaders(
        dataset_cfg=dataset_cfg,
        batch_size=train_cfg.batch_size,
        dataset_key=dataset_key,
        input_steps=input_steps,
        output_steps=output_steps,
        stride=dataset_cfg.get("stride", 1),
        cache=dataset_cfg.get("cache", "none"),
        num_workers=train_cfg.num_workers,
        pin_memory=train_cfg.pin_memory,
        persistent_workers=train_cfg.persistent_workers,
        prefetch_factor=train_cfg.prefetch_factor,
    )
    train_loader = loaders["train"]

    in_channels = input_steps * fields
    out_channels = output_steps * fields
    model = build_model(config.get("model", {}), in_channels, out_channels).to(device)
    param_count = count_parameters(model)
    optimizer = AdamW(model.parameters(), lr=train_cfg.lr)
    scheduler = StepLR(optimizer, step_size=train_cfg.lr_step, gamma=train_cfg.lr_gamma)
    use_amp = train_cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    if train_cfg.checkpoint_every > 0 or train_cfg.checkpoint_best_metric:
        if train_cfg.checkpoint_dir:
            ckpt_dir = Path(train_cfg.checkpoint_dir)
        else:
            ckpt_dir = Path("outputs/checkpoints") / dataset_cfg.get("name", "run") / config.get("model", {}).get("name", "model")
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    else:
        ckpt_dir = None
    best_metric_value = float("inf")
    best_metric_key = train_cfg.checkpoint_best_metric

    start_epoch = 0
    if train_cfg.resume_from:
        resume_path = Path(train_cfg.resume_from)
        start_epoch = load_checkpoint(resume_path, model, optimizer, scheduler, device)
        if start_epoch < 0:
            start_epoch = 0

    logger = MetricsLogger(Path(config.get("logging", {}).get("metrics_path", "outputs/logs/metrics.jsonl")))

    epoch_bar = tqdm(
        range(start_epoch, train_cfg.epochs),
        desc="epochs",
        initial=start_epoch,
        total=train_cfg.epochs,
        dynamic_ncols=True,
    )
    for epoch in epoch_bar:
        loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            output_steps,
            fields,
            scaler=scaler,
            amp=use_amp,
        )
        log_payload = {"loss": loss, "params": param_count}
        if train_cfg.eval_every > 0 and (epoch + 1) % train_cfg.eval_every == 0:
            for split in eval_splits:
                loader = loaders.get(split)
                if loader is None:
                    continue
                metrics = evaluate(
                    model,
                    loader,
                    device,
                    output_steps,
                    fields,
                    windows=eval_windows,
                    amp=use_amp,
                )
                for key, value in metrics.items():
                    log_payload[f"{split}_{key}"] = value
        logger.log(epoch, log_payload)
        epoch_bar.set_postfix(
            {k: f"{v:.4e}" for k, v in log_payload.items() if isinstance(v, float)}
        )
        if ckpt_dir is not None:
            if train_cfg.checkpoint_every > 0 and (epoch + 1) % train_cfg.checkpoint_every == 0:
                save_checkpoint(model, optimizer, epoch + 1, ckpt_dir / "last.pt", scheduler=scheduler)
            if best_metric_key and best_metric_key in log_payload:
                current = log_payload[best_metric_key]
                if isinstance(current, float) and current < best_metric_value:
                    best_metric_value = current
                    save_checkpoint(model, optimizer, epoch + 1, ckpt_dir / "best.pt", scheduler=scheduler)
        scheduler.step()
    if train_cfg.test_at_end and "test" in loaders:
        test_metrics = evaluate(
            model,
            loaders["test"],
            device,
            output_steps,
            fields,
            windows=eval_windows,
            amp=use_amp,
        )
        logger.log(train_cfg.epochs, {f"test_{key}": value for key, value in test_metrics.items()})
