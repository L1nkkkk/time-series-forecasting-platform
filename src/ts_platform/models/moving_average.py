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
        num_features: int,
        window_size: int | None = None,
    ) -> None:
        super().__init__(input_len, output_len, num_features)
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

        if x.ndim != 3:
            msg = "x must be shaped [batch, input_len, num_features]"
            raise ValueError(msg)
        average = x[:, -self.window_size :, :].mean(dim=1, keepdim=True)
        return average.repeat(1, self.output_len, 1)


if "moving_average" not in MODEL_REGISTRY.names():
    MODEL_REGISTRY.register("moving_average", MovingAverageForecastModel)
