"""Temporal Convolutional Network (TCN) placeholder."""

from torch import nn


class WildfireTCN(nn.Module):
    """Stub TCN model."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Identity()

    def forward(self, sequence):
        return self.backbone(sequence)
