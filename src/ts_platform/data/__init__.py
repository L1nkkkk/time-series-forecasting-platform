"""Dataset abstractions and built-in dataset registrations."""

from ts_platform.data.base import ForecastBatch, ForecastingDataset
from ts_platform.data.catalog import DATASET_CATALOG, DatasetMetadata
from ts_platform.data.catalog_loader import load_dataset_catalog, register_dataset_catalog
from ts_platform.data.loaders import build_dataset
from ts_platform.data.profile import DatasetProfile, profile_csv_dataset
from ts_platform.data.registry import DATASET_REGISTRY

__all__ = [
    "DATASET_CATALOG",
    "DATASET_REGISTRY",
    "DatasetMetadata",
    "DatasetProfile",
    "ForecastBatch",
    "ForecastingDataset",
    "build_dataset",
    "load_dataset_catalog",
    "profile_csv_dataset",
    "register_dataset_catalog",
]
