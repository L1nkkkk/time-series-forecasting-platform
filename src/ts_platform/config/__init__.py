"""Configuration loading and validation."""

from ts_platform.config.loader import load_config, save_config_snapshot
from ts_platform.config.schema import PlatformConfig

__all__ = ["PlatformConfig", "load_config", "save_config_snapshot"]
