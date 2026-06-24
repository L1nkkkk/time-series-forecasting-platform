"""Lightweight PatchTST-style forecasting model."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn

from ts_platform.models.base import BaseForecastModel
from ts_platform.models.registry import MODEL_REGISTRY


class PatchTSTForecastModel(BaseForecastModel):
    """Channel-independent patch Transformer for direct multi-step forecasting."""

    def __init__(
        self,
        input_len: int,
        output_len: int,
        num_features: int | None = None,
        *,
        patch_len: int = 8,
        stride: int = 4,
        d_model: int = 32,
        num_heads: int = 4,
        num_layers: int = 1,
        dim_feedforward: int = 64,
        dropout: float = 0.0,
        input_dim: int | None = None,
        target_dim: int | None = None,
    ) -> None:
        super().__init__(
            input_len,
            output_len,
            num_features,
            input_dim=input_dim,
            target_dim=target_dim,
        )
        if patch_len <= 0:
            msg = "patch_len must be positive"
            raise ValueError(msg)
        if stride <= 0:
            msg = "stride must be positive"
            raise ValueError(msg)
        if patch_len > input_len:
            msg = "patch_len must be <= input_len"
            raise ValueError(msg)
        if d_model <= 0:
            msg = "d_model must be positive"
            raise ValueError(msg)
        if num_heads <= 0:
            msg = "num_heads must be positive"
            raise ValueError(msg)
        if d_model % num_heads != 0:
            msg = "d_model must be divisible by num_heads"
            raise ValueError(msg)
        if num_layers <= 0:
            msg = "num_layers must be positive"
            raise ValueError(msg)
        if dim_feedforward <= 0:
            msg = "dim_feedforward must be positive"
            raise ValueError(msg)
        if dropout < 0 or dropout >= 1:
            msg = "dropout must be >= 0 and < 1"
            raise ValueError(msg)

        self.patch_len = patch_len
        self.stride = stride
        self.num_patches = 1 + (input_len - patch_len) // stride
        self.patch_projection = nn.Linear(patch_len, d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(self.num_patches * d_model, output_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode target-channel patches and return ``[batch, output_len, target_dim]``."""

        target_x = self.target_slice(x).transpose(1, 2)
        batch_size, target_dim, _ = target_x.shape
        patches = target_x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        patches = patches.reshape(batch_size * target_dim, self.num_patches, self.patch_len)
        encoded = self.patch_projection(patches) + self.position_embedding
        memory = self.encoder(encoded)
        output = self.head(memory.reshape(batch_size * target_dim, -1))
        output = output.reshape(batch_size, target_dim, self.output_len).transpose(1, 2)
        return cast(torch.Tensor, output)


if "patchtst" not in MODEL_REGISTRY.names():
    MODEL_REGISTRY.register("patchtst", PatchTSTForecastModel)
