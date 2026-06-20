"""Moving-average forecasting baseline."""

from __future__ import annotations

import torch

from ts_platform.models.base import BaseForecastModel
from ts_platform.models.registry import MODEL_REGISTRY


class MovingAverageForecastModel(BaseForecastModel):
    """Repeat the mean of the last window of history."""

    def __init__(
        self,
        input_len: int,
        output_len: int,
        num_features: int | None = None,
        window_size: int | None = None,
        *,
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
        resolved_window_size = input_len if window_size is None else window_size
        if resolved_window_size <= 0:
            msg = "window_size must be positive"
            raise ValueError(msg)
        if resolved_window_size > input_len:
            msg = "window_size cannot be greater than input_len"
            raise ValueError(msg)
        self.window_size = resolved_window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the moving average repeated over the forecast horizon."""

        target_x = self.target_slice(x)
        average = target_x[:, -self.window_size :, :].mean(dim=1, keepdim=True)
        return average.repeat(1, self.output_len, 1)


if "moving_average" not in MODEL_REGISTRY.names():
    MODEL_REGISTRY.register("moving_average", MovingAverageForecastModel)
