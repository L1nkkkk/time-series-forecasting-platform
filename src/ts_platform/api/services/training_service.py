"""Training service used by API routes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ts_platform.config.schema import PlatformConfig
from ts_platform.runner.trainer import Trainer


def train_with_safe_output_dir(config: PlatformConfig, *, runs_root: Path) -> dict[str, Any]:
    """Run training after forcing the experiment output root to the API-safe root."""

    safe_experiment = config.experiment.model_copy(update={"output_dir": runs_root})
    safe_config = config.model_copy(deep=True, update={"experiment": safe_experiment})
    result = Trainer(safe_config).run()
    return result.to_dict()
