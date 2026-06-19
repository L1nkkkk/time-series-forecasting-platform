"""Min-max scaler."""

from __future__ import annotations

import torch

from ts_platform.scaler.base import BaseScaler


class MinMaxScaler(BaseScaler):
    """Feature-wise min-max scaler."""

    def __init__(self, feature_range: tuple[float, float] = (0.0, 1.0), eps: float = 1e-8) -> None:
        low, high = feature_range
        if high <= low:
            msg = "feature_range high must be greater than low"
            raise ValueError(msg)
        self.feature_range = feature_range
        self.eps = eps
        self.fitted = False
        self.data_min: torch.Tensor | None = None
        self.data_max: torch.Tensor | None = None

    def fit(self, values: torch.Tensor) -> MinMaxScaler:
        """Fit feature-wise min and max."""

        self._validate_values(values)
        dims = tuple(range(values.ndim - 1))
        self.data_min = values.amin(dim=dims, keepdim=True)
        self.data_max = values.amax(dim=dims, keepdim=True)
        self.fitted = True
        return self

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        """Apply min-max scaling."""

        data_min, scale = self._state()
        low, high = self.feature_range
        normalized = (values - data_min.to(values.device)) / scale.to(values.device)
        return normalized * (high - low) + low

    def inverse_transform(self, values: torch.Tensor) -> torch.Tensor:
        """Undo min-max scaling."""

        data_min, scale = self._state()
        low, high = self.feature_range
        normalized = (values - low) / (high - low)
        return normalized * scale.to(values.device) + data_min.to(values.device)

    def _state(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.fitted or self.data_min is None or self.data_max is None:
            msg = "MinMaxScaler must be fitted before use"
            raise RuntimeError(msg)
        scale = (self.data_max - self.data_min).clamp_min(self.eps)
        return self.data_min, scale
