"""Dataset abstractions and built-in dataset registrations."""

from ts_platform.data.assets import (
    clear_dataset_asset,
    dataset_asset_status,
    list_dataset_cache,
    prepare_dataset_asset,
    prepared_metadata_from_status,
)
from ts_platform.data.base import ForecastBatch, ForecastDimensions, ForecastingDataset
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
    "ForecastDimensions",
    "ForecastingDataset",
    "build_dataset",
    "clear_dataset_asset",
    "dataset_asset_status",
    "list_dataset_cache",
    "load_dataset_catalog",
    "prepare_dataset_asset",
    "prepared_metadata_from_status",
    "profile_csv_dataset",
    "register_dataset_catalog",
]
