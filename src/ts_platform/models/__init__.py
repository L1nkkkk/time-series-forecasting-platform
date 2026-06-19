"""Forecasting models and registry."""

from ts_platform.models.base import BaseForecastModel
from ts_platform.models.linear import LinearForecastModel
from ts_platform.models.mlp import MLPForecastModel
from ts_platform.models.moving_average import MovingAverageForecastModel
from ts_platform.models.naive import NaiveLastValueModel
from ts_platform.models.registry import MODEL_REGISTRY, build_model
from ts_platform.models.seasonal_naive import SeasonalNaiveForecastModel

__all__ = [
    "MODEL_REGISTRY",
    "BaseForecastModel",
    "LinearForecastModel",
    "MLPForecastModel",
    "MovingAverageForecastModel",
    "NaiveLastValueModel",
    "SeasonalNaiveForecastModel",
    "build_model",
]
