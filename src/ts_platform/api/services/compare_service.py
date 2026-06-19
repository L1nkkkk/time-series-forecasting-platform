"""Compare service used by API routes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ts_platform.config.compare_schema import CompareConfig
from ts_platform.runner.comparer import CompareRunner


def compare_with_safe_output_dir(config: CompareConfig, *, runs_root: Path) -> dict[str, Any]:
    """Run compare after forcing the experiment output root to the API-safe root."""

    safe_experiment = config.experiment.model_copy(update={"output_dir": runs_root})
    safe_config = config.model_copy(deep=True, update={"experiment": safe_experiment})
    result = CompareRunner(safe_config).run()
    return result.to_dict()
