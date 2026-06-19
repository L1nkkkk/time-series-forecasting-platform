"""Reproducibility helpers."""

from __future__ import annotations

import importlib.metadata
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_environment(cwd: Path | None = None) -> dict[str, Any]:
    """Collect runtime environment metadata for reproducibility."""

    working_dir = cwd or Path.cwd()
    packages: dict[str, str | None] = {}
    for package in ["torch", "numpy", "pandas", "pydantic", "PyYAML", "fastapi"]:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "git_commit": _git_commit(working_dir),
    }


def _git_commit(cwd: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None
