from __future__ import annotations

import torch
from torch import nn


def compl_mul2d(input_tensor: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bixy,ioxy->boxy", input_tensor, weights)


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batchsize, _, height, width = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            height,
            width // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, : self.modes1, : self.modes2] = compl_mul2d(
            x_ft[:, :, : self.modes1, : self.modes2],
            self.weights,
        )
        x = torch.fft.irfft2(out_ft, s=(height, width), norm="ortho")
        return x


class FNOS(nn.Module):
    """Fourier Neural Operator (small) skeleton."""

    def __init__(self, in_channels: int, out_channels: int, width: int = 64, modes: int = 12, layers: int = 8):
        super().__init__()
        self.width = width
        self.input_proj = nn.Conv2d(in_channels, width, kernel_size=1)
        self.spectral_layers = nn.ModuleList(
            [SpectralConv2d(width, width, modes, modes) for _ in range(layers)]
        )
        self.pointwise_layers = nn.ModuleList(
            [nn.Conv2d(width, width, kernel_size=1) for _ in range(layers)]
        )
        self.act = nn.GELU()
        self.output_proj = nn.Conv2d(width, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for spectral, pointwise in zip(self.spectral_layers, self.pointwise_layers):
            x = self.act(spectral(x) + pointwise(x))
        return self.output_proj(x)
