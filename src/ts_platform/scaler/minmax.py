"""Min-max scaler."""

from __future__ import annotations

from typing import Any

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

    def state_dict(self) -> dict[str, Any]:
        """Return serializable scaler state."""

        return {
            "fitted": self.fitted,
            "data_min": self.data_min,
            "data_max": self.data_max,
            "feature_range": self.feature_range,
            "eps": self.eps,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore scaler state."""

        fitted = state.get("fitted")
        data_min = state.get("data_min")
        data_max = state.get("data_max")
        feature_range = state.get("feature_range")
        eps = state.get("eps")
        if not isinstance(fitted, bool):
            msg = "MinMaxScaler state field 'fitted' must be a bool"
            raise ValueError(msg)
        if not isinstance(eps, int | float):
            msg = "MinMaxScaler state field 'eps' must be numeric"
            raise ValueError(msg)
        if not isinstance(feature_range, tuple | list) or len(feature_range) != 2:
            msg = "MinMaxScaler state field 'feature_range' must contain two values"
            raise ValueError(msg)
        low, high = float(feature_range[0]), float(feature_range[1])
        if high <= low:
            msg = "MinMaxScaler feature_range high must be greater than low"
            raise ValueError(msg)
        if fitted and not isinstance(data_min, torch.Tensor):
            msg = "MinMaxScaler fitted state requires tensor field 'data_min'"
            raise ValueError(msg)
        if fitted and not isinstance(data_max, torch.Tensor):
            msg = "MinMaxScaler fitted state requires tensor field 'data_max'"
            raise ValueError(msg)
        self.fitted = fitted
        self.data_min = data_min if isinstance(data_min, torch.Tensor) else None
        self.data_max = data_max if isinstance(data_max, torch.Tensor) else None
        self.feature_range = (low, high)
        self.eps = float(eps)

    def _state(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.fitted or self.data_min is None or self.data_max is None:
            msg = "MinMaxScaler must be fitted before use"
            raise RuntimeError(msg)
        scale = (self.data_max - self.data_min).clamp_min(self.eps)
        return self.data_min, scale
