from __future__ import annotations

from typing import Iterable

import numpy as np
import torch


def _to_tensor(array):
    if isinstance(array, torch.Tensor):
        return array
    return torch.from_numpy(np.asarray(array))


def rmse(pred, target, dim=None):
    pred_t = _to_tensor(pred).float()
    target_t = _to_tensor(target).float()
    error = pred_t - target_t
    mse = (error ** 2).mean(dim=dim)
    return torch.sqrt(mse)


def vrmse(pred, target, dim=None, eps: float = 1e-8):
    pred_t = _to_tensor(pred).float()
    target_t = _to_tensor(target).float()
    error = pred_t - target_t
    mse = (error ** 2).mean(dim=dim)
    variance = target_t.var(dim=dim, unbiased=False)
    return torch.sqrt(mse / (variance + eps))


def window_vrmse(pred, target, start: int, end: int, time_dim: int = 1) -> torch.Tensor:
    pred_t = _to_tensor(pred)
    target_t = _to_tensor(target)
    slicer = [slice(None)] * pred_t.ndim
    slicer[time_dim] = slice(start, end)
    pred_slice = pred_t[tuple(slicer)]
    target_slice = target_t[tuple(slicer)]
    return vrmse(pred_slice, target_slice)


def windowed_vrmse(pred, target, windows: Iterable[tuple[int, int]] = ((6, 12), (13, 30)), time_dim: int = 1):
    metrics = {}
    for start, end in windows:
        metrics[f"vrmse_{start}_{end}"] = window_vrmse(pred, target, start, end, time_dim=time_dim)
    return metrics
