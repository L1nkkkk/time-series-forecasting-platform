"""Scaler base classes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch


class BaseScaler(ABC):
    """Base interface for tensor scalers."""

    fitted: bool

    @abstractmethod
    def fit(self, values: torch.Tensor) -> BaseScaler:
        """Fit scaler state from values."""

    @abstractmethod
    def transform(self, values: torch.Tensor) -> torch.Tensor:
        """Scale values."""

    @abstractmethod
    def inverse_transform(self, values: torch.Tensor) -> torch.Tensor:
        """Undo scaling."""

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        """Return serializable scaler state."""

    @abstractmethod
    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore scaler state."""

    @staticmethod
    def _validate_values(values: torch.Tensor) -> None:
        if values.numel() == 0:
            msg = "cannot scale an empty tensor"
            raise ValueError(msg)
        if values.ndim < 1:
            msg = "values must have at least one dimension"
            raise ValueError(msg)


class IdentityScaler(BaseScaler):
    """No-op scaler for explicit opt-out."""

    def __init__(self) -> None:
        self.fitted = True

    def fit(self, values: torch.Tensor) -> IdentityScaler:
        """Validate values and keep no state."""

        self._validate_values(values)
        return self

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        """Return values unchanged."""

        return values

    def inverse_transform(self, values: torch.Tensor) -> torch.Tensor:
        """Return values unchanged."""

        return values

    def state_dict(self) -> dict[str, Any]:
        """Return no-op scaler state."""

        return {"fitted": self.fitted}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore no-op scaler state."""

        fitted = state.get("fitted", True)
        if not isinstance(fitted, bool):
            msg = "IdentityScaler state field 'fitted' must be a bool"
            raise ValueError(msg)
        self.fitted = fitted
