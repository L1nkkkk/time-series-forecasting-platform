"""Recurrent forecasting baselines."""

from __future__ import annotations

from typing import ClassVar, TypeAlias, cast

import torch
from torch import nn

from ts_platform.models.base import BaseForecastModel
from ts_platform.models.registry import MODEL_REGISTRY

RecurrentClass: TypeAlias = type[nn.RNN] | type[nn.GRU] | type[nn.LSTM]


class _RecurrentForecastModel(BaseForecastModel):
    """Shared direct multi-step forecaster for recurrent encoders."""

    recurrent_cls: ClassVar[RecurrentClass]

    def __init__(
        self,
        input_len: int,
        output_len: int,
        num_features: int,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ) -> None:
        super().__init__(input_len, output_len, num_features)
        if hidden_size <= 0:
            msg = "hidden_size must be positive"
            raise ValueError(msg)
        if num_layers <= 0:
            msg = "num_layers must be positive"
            raise ValueError(msg)
        if dropout < 0 or dropout >= 1:
            msg = "dropout must be >= 0 and < 1"
            raise ValueError(msg)

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.recurrent: nn.RNN | nn.GRU | nn.LSTM = self.recurrent_cls(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
            bidirectional=bidirectional,
        )
        projection_input_size = hidden_size * (2 if bidirectional else 1)
        self.projection = nn.Linear(projection_input_size, output_len * num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode the history and project the final hidden state to the horizon."""

        if x.ndim != 3:
            msg = "x must be shaped [batch, input_len, num_features]"
            raise ValueError(msg)
        batch_size = x.shape[0]
        _, hidden = self.recurrent(x)
        hidden_state = hidden[0] if isinstance(hidden, tuple) else hidden
        final_hidden = self._final_layer_hidden(hidden_state, batch_size)
        output = self.projection(final_hidden)
        return cast(torch.Tensor, output.reshape(batch_size, self.output_len, self.num_features))

    def _final_layer_hidden(self, hidden_state: torch.Tensor, batch_size: int) -> torch.Tensor:
        if not self.bidirectional:
            return hidden_state[-1]
        hidden_by_layer = hidden_state.reshape(self.num_layers, 2, batch_size, self.hidden_size)
        forward_hidden = hidden_by_layer[-1, 0]
        backward_hidden = hidden_by_layer[-1, 1]
        return torch.cat((forward_hidden, backward_hidden), dim=1)


class RNNForecastModel(_RecurrentForecastModel):
    """Vanilla RNN direct multi-step forecasting baseline."""

    recurrent_cls = nn.RNN


class GRUForecastModel(_RecurrentForecastModel):
    """GRU direct multi-step forecasting baseline."""

    recurrent_cls = nn.GRU


class LSTMForecastModel(_RecurrentForecastModel):
    """LSTM direct multi-step forecasting baseline."""

    recurrent_cls = nn.LSTM


if "rnn" not in MODEL_REGISTRY.names():
    MODEL_REGISTRY.register("rnn", RNNForecastModel)
if "gru" not in MODEL_REGISTRY.names():
    MODEL_REGISTRY.register("gru", GRUForecastModel)
if "lstm" not in MODEL_REGISTRY.names():
    MODEL_REGISTRY.register("lstm", LSTMForecastModel)
