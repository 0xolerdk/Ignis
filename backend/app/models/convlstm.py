"""ConvLSTM architecture for wildfire nowcasting."""

from __future__ import annotations

from typing import List, Sequence, Tuple, Union

import torch
from torch import nn


class ConvLSTMCell(nn.Module):
    """Basic ConvLSTM cell."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_size: int,
        bias: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout_layer = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        self.conv = nn.Conv2d(
            input_dim + hidden_dim,
            4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )

    def forward(
        self,
        input_tensor: torch.Tensor,
        cur_state: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined = self.dropout_layer(combined)
        gates = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(
        self, batch_size: int, spatial_size: Tuple[int, int], device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        height, width = spatial_size
        h = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        return h, c


class ConvLSTM(nn.Module):
    """Stacked ConvLSTM implementation."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        kernel_size: int = 3,
        bias: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one dimension.")
        self.input_dim = input_dim
        self.hidden_dims = list(hidden_dims)
        self.num_layers = len(hidden_dims)
        self.cell_list = nn.ModuleList()
        for i in range(self.num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dims[i - 1]
            cell = ConvLSTMCell(
                input_dim=cur_input_dim,
                hidden_dim=hidden_dims[i],
                kernel_size=kernel_size,
                bias=bias,
                dropout=dropout,
            )
            self.cell_list.append(cell)

    def forward(
        self, input_tensor: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            input_tensor: shape (batch, seq_len, channels, height, width)
        Returns:
            layer_outputs: list over layers -> tensor (batch, seq_len, hidden_dim, height, width)
            last_states: list of (h, c) for each layer
        """
        batch, seq_len, _, height, width = input_tensor.shape
        device = input_tensor.device
        cur_input = input_tensor

        layer_outputs: List[torch.Tensor] = []
        last_states: List[Tuple[torch.Tensor, torch.Tensor]] = []

        for layer_idx, cell in enumerate(self.cell_list):
            h, c = cell.init_hidden(batch, (height, width), device)
            output_inner = []
            for t in range(seq_len):
                h, c = cell(cur_input[:, t, :, :, :], (h, c))
                output_inner.append(h)
            layer_output = torch.stack(output_inner, dim=1)
            layer_outputs.append(layer_output)
            last_states.append((h, c))
            cur_input = layer_output

        return layer_outputs, last_states


class WildfireConvLSTM(nn.Module):
    """ConvLSTM-based decoder producing multi-horizon nowcasts."""

    def __init__(
        self,
        *,
        in_channels: int,
        hidden_dims: Sequence[int],
        horizons: Sequence[int],
        kernel_size: int = 3,
        dropout: float = 0.2,
        bilinear_head: bool = True,
    ) -> None:
        super().__init__()
        self.horizons = list(horizons)
        self.backbone = ConvLSTM(
            input_dim=in_channels,
            hidden_dims=hidden_dims,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        final_dim = hidden_dims[-1]
        self.dropout = nn.Dropout2d(dropout)
        self.head: Union[nn.Sequential, nn.Conv2d]
        if bilinear_head:
            self.head = nn.Sequential(
                nn.Conv2d(final_dim, final_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(final_dim),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout),
                nn.Conv2d(final_dim, len(self.horizons), kernel_size=1),
            )
        else:
            self.head = nn.Conv2d(final_dim, len(self.horizons), kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor of shape (batch, seq_len, channels, height, width)
        Returns:
            logits of shape (batch, horizons, height, width)
        """
        layer_outputs, _ = self.backbone(x)
        last_output = layer_outputs[-1][:, -1]
        last_output = self.dropout(last_output)
        return self.head(last_output)
