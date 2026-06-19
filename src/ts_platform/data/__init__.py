"""Dataset abstractions and built-in dataset registrations."""

from ts_platform.data.base import ForecastBatch, ForecastingDataset
from ts_platform.data.catalog import DATASET_CATALOG, DatasetMetadata
from ts_platform.data.loaders import build_dataset
from ts_platform.data.registry import DATASET_REGISTRY

__all__ = [
    "DATASET_CATALOG",
    "DATASET_REGISTRY",
    "DatasetMetadata",
    "ForecastBatch",
    "ForecastingDataset",
    "build_dataset",
]
