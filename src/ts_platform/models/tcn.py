"""Temporal convolutional forecasting baseline."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn

from ts_platform.models.base import BaseForecastModel
from ts_platform.models.registry import MODEL_REGISTRY


class _CausalConvBlock(nn.Module):
    """Small causal-ish Conv1d block that preserves sequence length."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.crop = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.crop,
        )
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution, crop right padding, and activate."""

        output = self.conv(x)
        if self.crop > 0:
            output = output[:, :, : -self.crop]
        output = self.activation(output)
        return cast(torch.Tensor, self.dropout(output))


class TCNForecastModel(BaseForecastModel):
    """Lightweight TCN direct multi-step forecasting baseline."""

    def __init__(
        self,
        input_len: int,
        output_len: int,
        num_features: int,
        hidden_channels: int = 32,
        num_layers: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(input_len, output_len, num_features)
        if hidden_channels <= 0:
            msg = "hidden_channels must be positive"
            raise ValueError(msg)
        if num_layers <= 0:
            msg = "num_layers must be positive"
            raise ValueError(msg)
        if kernel_size <= 0:
            msg = "kernel_size must be positive"
            raise ValueError(msg)
        if dropout < 0 or dropout >= 1:
            msg = "dropout must be >= 0 and < 1"
            raise ValueError(msg)

        layers: list[nn.Module] = []
        in_channels = num_features
        for layer_index in range(num_layers):
            dilation = 2**layer_index
            layers.append(
                _CausalConvBlock(
                    in_channels,
                    hidden_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            in_channels = hidden_channels
        self.network = nn.Sequential(*layers)
        self.projection = nn.Linear(hidden_channels, output_len * num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run TCN encoder and project the final hidden step to the horizon."""

        if x.ndim != 3:
            msg = "x must be shaped [batch, input_len, num_features]"
            raise ValueError(msg)
        batch_size = x.shape[0]
        features_first = x.transpose(1, 2)
        encoded = self.network(features_first)
        final_step = encoded[:, :, -1]
        output = self.projection(final_step)
        return cast(torch.Tensor, output.reshape(batch_size, self.output_len, self.num_features))


if "tcn" not in MODEL_REGISTRY.names():
    MODEL_REGISTRY.register("tcn", TCNForecastModel)
