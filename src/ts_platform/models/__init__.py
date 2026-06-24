"""Forecasting models and registry."""

from ts_platform.models.base import BaseForecastModel
from ts_platform.models.linear import LinearForecastModel
from ts_platform.models.linear_family import DLinearForecastModel, NLinearForecastModel
from ts_platform.models.mlp import MLPForecastModel
from ts_platform.models.moving_average import MovingAverageForecastModel
from ts_platform.models.naive import NaiveLastValueModel
from ts_platform.models.nbeats import NBeatsForecastModel
from ts_platform.models.patchtst import PatchTSTForecastModel
from ts_platform.models.recurrent import GRUForecastModel, LSTMForecastModel, RNNForecastModel
from ts_platform.models.registry import MODEL_REGISTRY, build_model
from ts_platform.models.seasonal_naive import SeasonalNaiveForecastModel
from ts_platform.models.tcn import TCNForecastModel
from ts_platform.models.transformer import TransformerForecastModel

__all__ = [
    "MODEL_REGISTRY",
    "BaseForecastModel",
    "DLinearForecastModel",
    "GRUForecastModel",
    "LSTMForecastModel",
    "LinearForecastModel",
    "MLPForecastModel",
    "MovingAverageForecastModel",
    "NBeatsForecastModel",
    "NaiveLastValueModel",
    "NLinearForecastModel",
    "PatchTSTForecastModel",
    "RNNForecastModel",
    "SeasonalNaiveForecastModel",
    "TCNForecastModel",
    "TransformerForecastModel",
    "build_model",
]
