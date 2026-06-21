"""Logging helpers for experiments."""

from __future__ import annotations

import logging
from pathlib import Path


def setup_experiment_logger(run_dir: Path) -> logging.Logger:
    """Create a logger that writes to stdout and train.log."""

    logger_name = _experiment_logger_name(run_dir)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(run_dir / "train.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def close_experiment_logger_for_run_dir(run_dir: Path) -> None:
    """Close any cached handlers for a run directory logger."""

    logger = logging.getLogger(_experiment_logger_name(run_dir))
    for handler in list(logger.handlers):
        handler.flush()
        handler.close()
        logger.removeHandler(handler)


def _experiment_logger_name(run_dir: Path) -> str:
    return f"ts_platform.{run_dir.as_posix()}"
