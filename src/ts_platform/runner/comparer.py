"""Multi-model compare runner."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from ts_platform.config.compare_loader import load_compare_config, save_compare_config_snapshot
from ts_platform.config.compare_schema import CompareConfig, CompareModelConfig
from ts_platform.config.schema import ExperimentConfig, ModelConfig, PlatformConfig
from ts_platform.experiment.artifacts import build_compare_artifact_manifest, save_artifact_manifest
from ts_platform.experiment.recorder import ExperimentRecorder
from ts_platform.experiment.reproducibility import collect_environment
from ts_platform.runner.trainer import Trainer, TrainingResult

CompareStatus = Literal["success", "failed"]


@dataclass(frozen=True)
class CompareModelResult:
    """Result for one model inside a compare run."""

    model_name: str
    model_alias: str
    model_params: dict[str, Any]
    status: CompareStatus
    run_id: str | None
    run_dir: Path | None
    checkpoint_path: Path | None
    test_metrics: dict[str, float] | None
    data_metadata: dict[str, Any] | None = None
    created_at: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class CompareResult:
    """Serializable compare result summary."""

    experiment_name: str
    compare_run_dir: Path
    compare_run_id: str
    created_at: str
    leaderboard_json_path: Path
    leaderboard_csv_path: Path
    primary_metric: str
    success_count: int
    failed_count: int
    rows: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable result."""

        return {
            "run_type": "compare",
            "experiment_name": self.experiment_name,
            "compare_run_dir": str(self.compare_run_dir),
            "compare_run_id": self.compare_run_id,
            "created_at": self.created_at,
            "leaderboard_json_path": str(self.leaderboard_json_path),
            "leaderboard_csv_path": str(self.leaderboard_csv_path),
            "primary_metric": self.primary_metric,
            "success_count": self.success_count,
            "failed_count": self.failed_count,
            "rows": self.rows,
        }


class CompareRunner:
    """Run multiple models through the existing Trainer and build a leaderboard."""

    def __init__(self, config: CompareConfig) -> None:
        self.config = config

    @classmethod
    def from_config_path(cls, path: str | Path) -> CompareRunner:
        """Build a compare runner from a YAML or JSON config file."""

        return cls(load_compare_config(path))

    def run(self) -> CompareResult:
        """Run all configured models and write compare artifacts."""

        recorder = ExperimentRecorder(
            self.config.experiment.output_dir,
            self.config.experiment.name,
            overwrite=self.config.experiment.overwrite,
        )
        compare_run_dir = recorder.prepare()
        save_compare_config_snapshot(self.config, compare_run_dir / "compare_config_snapshot.yaml")
        recorder.save_environment(collect_environment())

        model_results: list[CompareModelResult] = []
        total_models = len(self.config.models)
        self._write_progress(
            compare_run_dir,
            status="running",
            total_models=total_models,
            completed_results=model_results,
            current_model=None,
            current_model_alias=None,
        )
        for index, model_config in enumerate(self.config.models, start=1):
            model_alias = self._model_alias(index, model_config)
            self._write_progress(
                compare_run_dir,
                status="running",
                total_models=total_models,
                completed_results=model_results,
                current_model=model_config,
                current_model_alias=model_alias,
            )
            try:
                model_results.append(self._run_model(model_config, model_alias, compare_run_dir))
            except Exception as exc:
                if not self.config.continue_on_error:
                    msg = f"compare model {model_alias!r} failed: {exc}"
                    raise RuntimeError(msg) from exc
                model_results.append(
                    CompareModelResult(
                        model_name=model_config.name,
                        model_alias=model_alias,
                        model_params=dict(model_config.params),
                        status="failed",
                        run_id=None,
                        run_dir=None,
                        checkpoint_path=None,
                        test_metrics=None,
                        error=str(exc),
                    )
                )
            self._write_progress(
                compare_run_dir,
                status="running",
                total_models=total_models,
                completed_results=model_results,
                current_model=None,
                current_model_alias=None,
            )

        rows = self._leaderboard_rows(model_results)
        leaderboard_json_path = compare_run_dir / "leaderboard.json"
        leaderboard_csv_path = compare_run_dir / "leaderboard.csv"
        self._write_leaderboard_json(leaderboard_json_path, rows)
        self._write_leaderboard_csv(leaderboard_csv_path, rows)
        result = CompareResult(
            experiment_name=self.config.experiment.name,
            compare_run_dir=compare_run_dir,
            compare_run_id=recorder.run_id,
            created_at=recorder.created_at,
            leaderboard_json_path=leaderboard_json_path,
            leaderboard_csv_path=leaderboard_csv_path,
            primary_metric=self.config.primary_metric or self.config.evaluation.metrics[0],
            success_count=sum(row["status"] == "success" for row in rows),
            failed_count=sum(row["status"] == "failed" for row in rows),
            rows=rows,
        )
        recorder.save_results(result.to_dict())
        save_artifact_manifest(
            build_compare_artifact_manifest(
                experiment_name=result.experiment_name,
                compare_run_id=result.compare_run_id,
                compare_run_dir=result.compare_run_dir,
                leaderboard_json_path=result.leaderboard_json_path,
                leaderboard_csv_path=result.leaderboard_csv_path,
            ),
            compare_run_dir / "artifacts.json",
        )
        self._write_progress(
            compare_run_dir,
            status="succeeded",
            total_models=total_models,
            completed_results=model_results,
            current_model=None,
            current_model_alias=None,
        )
        return result

    def _run_model(
        self,
        model_config: CompareModelConfig,
        model_alias: str,
        compare_run_dir: Path,
    ) -> CompareModelResult:
        platform_config = PlatformConfig(
            experiment=ExperimentConfig(
                name=model_alias,
                output_dir=compare_run_dir / "models",
                seed=self.config.experiment.seed,
                overwrite=True,
            ),
            data=self.config.data.model_copy(deep=True),
            model=ModelConfig(name=model_config.name, params=dict(model_config.params)),
            training=self.config.training.model_copy(deep=True),
            evaluation=self.config.evaluation.model_copy(deep=True),
        )
        result = Trainer(platform_config).run()
        return self._model_result_from_training(model_config, model_alias, result)

    def _model_result_from_training(
        self,
        model_config: CompareModelConfig,
        model_alias: str,
        result: TrainingResult,
    ) -> CompareModelResult:
        original_metrics = result.test_metrics.get("original")
        if not isinstance(original_metrics, dict):
            msg = f"model {model_alias!r} did not produce original-scale test metrics"
            raise ValueError(msg)
        return CompareModelResult(
            model_name=model_config.name,
            model_alias=model_alias,
            model_params=dict(model_config.params),
            status="success",
            run_id=result.run_id,
            run_dir=result.run_dir,
            checkpoint_path=result.checkpoint_path,
            test_metrics={key: float(value) for key, value in original_metrics.items()},
            data_metadata=result.data_metadata,
            created_at=result.created_at,
        )

    def _leaderboard_rows(self, model_results: list[CompareModelResult]) -> list[dict[str, Any]]:
        rows = [self._row_from_result(result) for result in model_results]
        success_rows = [
            row
            for row in rows
            if row["status"] == "success" and row["primary_metric_value"] is not None
        ]
        failed_rows = [row for row in rows if row["status"] != "success"]
        success_rows.sort(key=lambda row: float(row["primary_metric_value"]))
        for rank, row in enumerate(success_rows, start=1):
            row["rank"] = rank
        return success_rows + failed_rows

    def _row_from_result(self, result: CompareModelResult) -> dict[str, Any]:
        metrics = result.test_metrics or {}
        primary_metric = self.config.primary_metric or self.config.evaluation.metrics[0]
        data_metadata = result.data_metadata or {}
        row: dict[str, Any] = {
            "rank": None,
            "status": result.status,
            "model_name": result.model_name,
            "model_alias": result.model_alias,
            "model_params": dict(result.model_params),
            "feature_aware": data_metadata.get("feature_aware"),
            "input_dim": data_metadata.get("input_dim"),
            "target_dim": data_metadata.get("target_dim"),
            "feature_dim": data_metadata.get("feature_dim"),
            "target_cols": data_metadata.get("target_cols"),
            "feature_cols": data_metadata.get("feature_cols"),
            "run_id": result.run_id,
            "run_dir": str(result.run_dir) if result.run_dir is not None else None,
            "checkpoint_path": (
                str(result.checkpoint_path) if result.checkpoint_path is not None else None
            ),
            "primary_metric": primary_metric,
            "primary_metric_value": metrics.get(primary_metric),
            "created_at": result.created_at,
            "error": result.error,
        }
        for metric in self.config.evaluation.metrics:
            row[f"test_{metric}"] = metrics.get(metric)
        return row

    def _write_leaderboard_json(self, path: Path, rows: list[dict[str, Any]]) -> None:
        path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")

    def _write_leaderboard_csv(self, path: Path, rows: list[dict[str, Any]]) -> None:
        fieldnames = self._leaderboard_fieldnames()
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                csv_row = {field: row.get(field) for field in fieldnames}
                for json_field in ("model_params", "target_cols", "feature_cols"):
                    if isinstance(csv_row.get(json_field), (dict, list)):
                        csv_row[json_field] = json.dumps(
                            csv_row[json_field],
                            sort_keys=True,
                        )
                writer.writerow(csv_row)

    def _leaderboard_fieldnames(self) -> list[str]:
        return [
            "rank",
            "status",
            "model_name",
            "model_alias",
            "model_params",
            "feature_aware",
            "input_dim",
            "target_dim",
            "feature_dim",
            "target_cols",
            "feature_cols",
            "run_id",
            "run_dir",
            "checkpoint_path",
            "primary_metric",
            "primary_metric_value",
            "created_at",
            "error",
            *[f"test_{metric}" for metric in self.config.evaluation.metrics],
        ]

    def _model_alias(self, index: int, model_config: CompareModelConfig) -> str:
        alias = model_config.alias or model_config.name
        return f"{index:03d}_{alias}"

    def _write_progress(
        self,
        compare_run_dir: Path,
        *,
        status: str,
        total_models: int,
        completed_results: list[CompareModelResult],
        current_model: CompareModelConfig | None,
        current_model_alias: str | None,
    ) -> None:
        completed_models = len(completed_results)
        model_statuses = [self._progress_row_from_result(result) for result in completed_results]
        current_model_run_dir = None
        if current_model is not None and current_model_alias is not None:
            current_model_run_dir = compare_run_dir / "models" / current_model_alias / "latest"
            model_statuses.append(
                {
                    "model_name": current_model.name,
                    "model_alias": current_model_alias,
                    "status": "running",
                    "run_dir": str(current_model_run_dir),
                    "error": None,
                }
            )

        payload = {
            "status": status,
            "run_type": "compare",
            "experiment_name": self.config.experiment.name,
            "compare_run_dir": str(compare_run_dir),
            "updated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "total_models": total_models,
            "completed_models": completed_models,
            "progress_percent": round((completed_models / total_models) * 100, 2)
            if total_models
            else 100.0,
            "current_model": current_model.name if current_model is not None else None,
            "current_model_alias": current_model_alias,
            "current_model_run_dir": str(current_model_run_dir)
            if current_model_run_dir is not None
            else None,
            "model_statuses": model_statuses,
        }
        (compare_run_dir / "progress.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _progress_row_from_result(self, result: CompareModelResult) -> dict[str, Any]:
        return {
            "model_name": result.model_name,
            "model_alias": result.model_alias,
            "status": result.status,
            "run_id": result.run_id,
            "run_dir": str(result.run_dir) if result.run_dir is not None else None,
            "primary_metric": self.config.primary_metric or self.config.evaluation.metrics[0],
            "primary_metric_value": (result.test_metrics or {}).get(
                self.config.primary_metric or self.config.evaluation.metrics[0]
            ),
            "error": result.error,
        }
