"""U-Net architecture with dropout for wildfire segmentation."""

from __future__ import annotations

import torch
from torch import nn


class DoubleConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int | None = None,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, dropout: float, bilinear: bool
    ) -> None:
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(
                in_channels,
                out_channels,
                mid_channels=in_channels // 2,
                dropout=dropout,
            )
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels, dropout=dropout)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x1 = self.up(x1)
        diff_y = x2.size(-2) - x1.size(-2)
        diff_x = x2.size(-1) - x1.size(-1)
        if diff_x != 0 or diff_y != 0:
            x1 = nn.functional.pad(
                x1,
                [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
            )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.conv(x)


class WildfireUNet(nn.Module):
    """U-Net with MC-Dropout support."""

    def __init__(
        self,
        in_channels: int,
        base_filters: int = 32,
        dropout: float = 0.2,
        bilinear: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.base_filters = base_filters
        self.dropout_rate = dropout
        self.bilinear = bilinear

        factor = 2 if bilinear else 1

        self.inc = DoubleConv(in_channels, base_filters, dropout=dropout)
        self.down1 = Down(base_filters, base_filters * 2, dropout)
        self.down2 = Down(base_filters * 2, base_filters * 4, dropout)
        self.down3 = Down(base_filters * 4, base_filters * 8, dropout)
        self.down4 = Down(base_filters * 8, base_filters * 16 // factor, dropout)

        self.up1 = Up(base_filters * 16, base_filters * 8 // factor, dropout, bilinear)
        self.up2 = Up(base_filters * 8, base_filters * 4 // factor, dropout, bilinear)
        self.up3 = Up(base_filters * 4, base_filters * 2 // factor, dropout, bilinear)
        self.up4 = Up(base_filters * 2, base_filters, dropout, bilinear)
        self.outc = OutConv(base_filters, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)
