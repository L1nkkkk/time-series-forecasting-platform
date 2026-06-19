"""Standard score scaler."""

from __future__ import annotations

from typing import Any

import torch

from ts_platform.scaler.base import BaseScaler


class StandardScaler(BaseScaler):
    """Feature-wise z-score scaler."""

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = eps
        self.fitted = False
        self.mean: torch.Tensor | None = None
        self.std: torch.Tensor | None = None

    def fit(self, values: torch.Tensor) -> StandardScaler:
        """Fit feature-wise mean and standard deviation."""

        self._validate_values(values)
        dims = tuple(range(values.ndim - 1))
        self.mean = values.mean(dim=dims, keepdim=True)
        self.std = values.std(dim=dims, keepdim=True, unbiased=False).clamp_min(self.eps)
        self.fitted = True
        return self

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        """Apply z-score scaling."""

        mean, std = self._state()
        return (values - mean.to(values.device)) / std.to(values.device)

    def inverse_transform(self, values: torch.Tensor) -> torch.Tensor:
        """Undo z-score scaling."""

        mean, std = self._state()
        return values * std.to(values.device) + mean.to(values.device)

    def state_dict(self) -> dict[str, Any]:
        """Return serializable scaler state."""

        return {
            "fitted": self.fitted,
            "mean": self.mean,
            "std": self.std,
            "eps": self.eps,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore scaler state."""

        fitted = state.get("fitted")
        mean = state.get("mean")
        std = state.get("std")
        eps = state.get("eps")
        if not isinstance(fitted, bool):
            msg = "StandardScaler state field 'fitted' must be a bool"
            raise ValueError(msg)
        if not isinstance(eps, int | float):
            msg = "StandardScaler state field 'eps' must be numeric"
            raise ValueError(msg)
        if fitted and not isinstance(mean, torch.Tensor):
            msg = "StandardScaler fitted state requires tensor field 'mean'"
            raise ValueError(msg)
        if fitted and not isinstance(std, torch.Tensor):
            msg = "StandardScaler fitted state requires tensor field 'std'"
            raise ValueError(msg)
        self.fitted = fitted
        self.mean = mean if isinstance(mean, torch.Tensor) else None
        self.std = std if isinstance(std, torch.Tensor) else None
        self.eps = float(eps)

    def _state(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.fitted or self.mean is None or self.std is None:
            msg = "StandardScaler must be fitted before use"
            raise RuntimeError(msg)
        return self.mean, self.std
