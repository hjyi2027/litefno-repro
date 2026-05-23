from __future__ import annotations

import torch
from torch import nn


class LowRankBlock(nn.Module):
    def __init__(self, channels: int, rank: int):
        super().__init__()
        self.reduce = nn.Conv2d(channels, rank, kernel_size=1)
        self.conv = nn.Conv2d(rank, rank, kernel_size=3, padding=1)
        self.expand = nn.Conv2d(rank, channels, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act(self.reduce(x))
        x = self.act(self.conv(x))
        x = self.expand(x)
        return x + residual


class LiteFNO(nn.Module):
    """Lightweight FNO-style skeleton with low-rank bottlenecks."""

    def __init__(self, in_channels: int, out_channels: int, width: int = 64, rank: int = 32, layers: int = 8):
        super().__init__()
        self.input_proj = nn.Conv2d(in_channels, width, kernel_size=1)
        self.blocks = nn.ModuleList([LowRankBlock(width, rank) for _ in range(layers)])
        self.output_proj = nn.Conv2d(width, out_channels, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.input_proj(x))
        for block in self.blocks:
            x = block(x)
        return self.output_proj(x)
