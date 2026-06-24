"""Training runner implementation."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from ts_platform.config.loader import load_config
from ts_platform.config.schema import PlatformConfig
from ts_platform.data.base import ForecastBatch, ForecastingDataset
from ts_platform.data.loaders import build_dataset
from ts_platform.data.transforms import FeatureAwareScalerBundle, ScaledForecastingDataset
from ts_platform.experiment.artifacts import build_train_artifact_manifest, save_artifact_manifest
from ts_platform.experiment.logger import (
    close_experiment_logger_for_run_dir,
    setup_experiment_logger,
)
from ts_platform.experiment.recorder import ExperimentRecorder
from ts_platform.experiment.reproducibility import (
    build_worker_init_fn,
    collect_environment,
    set_seed,
)
from ts_platform.experiment.result_schema import MetricGroups, OptionalMetricGroups, result_payload
from ts_platform.models.registry import build_model
from ts_platform.runner.checkpoint import (
    CheckpointPayload,
    ScalerOrBundle,
    load_checkpoint,
    load_optimizer_state_from_checkpoint,
    restore_model_from_checkpoint,
    restore_scalers_from_checkpoint,
    save_checkpoint,
    validate_checkpoint_for_training,
)
from ts_platform.runner.devices import resolve_training_device
from ts_platform.runner.evaluator import collect_forecast_samples, evaluate
from ts_platform.runner.model_export import save_model_export
from ts_platform.scaler.base import BaseScaler
from ts_platform.scaler.registry import build_scaler


@dataclass(frozen=True)
class TrainingResult:
    """Serializable training result summary."""

    run_dir: Path
    run_id: str
    created_at: str
    experiment_name: str
    checkpoint_path: Path
    best_checkpoint_path: Path | None
    best_epoch: int | None
    best_metric: dict[str, Any] | None
    model_export_path: Path
    model_export_metadata_path: Path
    history: list[dict[str, Any]]
    validation_metrics: OptionalMetricGroups
    test_metrics: MetricGroups
    data_metadata: dict[str, Any]
    forecast_samples: dict[str, Any]
    resumed_from: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable result."""

        return result_payload(
            run_metadata={
                "run_id": self.run_id,
                "created_at": self.created_at,
                "run_dir": str(self.run_dir),
                "experiment_name": self.experiment_name,
            },
            checkpoint_path=str(self.checkpoint_path),
            best_checkpoint_path=(
                str(self.best_checkpoint_path) if self.best_checkpoint_path is not None else None
            ),
            best_epoch=self.best_epoch,
            best_metric=self.best_metric,
            model_export_path=str(self.model_export_path),
            model_export_metadata_path=str(self.model_export_metadata_path),
            history=self.history,
            validation_metrics=self.validation_metrics,
            test_metrics=self.test_metrics,
            data_metadata=self.data_metadata,
            forecast_samples=self.forecast_samples,
            resumed_from=str(self.resumed_from) if self.resumed_from is not None else None,
        )


class Trainer:
    """Configuration-driven training runner."""

    def __init__(self, config: PlatformConfig) -> None:
        self.config = config

    @classmethod
    def from_config_path(cls, path: str | Path) -> Trainer:
        """Build a trainer from a YAML or JSON config file."""

        return cls(load_config(path))

    def run(self) -> TrainingResult:
        """Run training, validation, testing, and artifact recording."""

        set_seed(self.config.experiment.seed)
        device = self._resolve_device()
        resume_checkpoint = self._load_resume_checkpoint()
        recorder = ExperimentRecorder(
            self.config.experiment.output_dir,
            self.config.experiment.name,
            overwrite=self.config.experiment.overwrite,
        )
        run_dir = recorder.prepare()
        logger = setup_experiment_logger(run_dir)
        try:
            return self._run_prepared(recorder, run_dir, logger, resume_checkpoint, device)
        finally:
            close_experiment_logger_for_run_dir(run_dir)

    def _run_prepared(
        self,
        recorder: ExperimentRecorder,
        run_dir: Path,
        logger: logging.Logger,
        resume_checkpoint: CheckpointPayload | None,
        device: torch.device,
    ) -> TrainingResult:
        """Run training after the run directory and logger are ready."""

        recorder.save_config(self.config)
        environment = collect_environment()
        recorder.save_environment(environment)

        logger.info("Using device: %s", device)

        train_dataset = build_dataset(self.config.data, "train", self.config.experiment.seed)
        val_dataset = build_dataset(self.config.data, "val", self.config.experiment.seed)
        test_dataset = build_dataset(self.config.data, "test", self.config.experiment.seed)

        if resume_checkpoint is not None:
            validate_checkpoint_for_training(
                resume_checkpoint,
                self.config,
                input_dim=train_dataset.input_dim,
                target_dim=train_dataset.target_dim,
                target_cols=_dataset_columns(train_dataset, "target_cols"),
                feature_cols=_dataset_columns(train_dataset, "feature_cols"),
            )
            scaler = restore_scalers_from_checkpoint(resume_checkpoint, self.config.data.scaler)
        else:
            scaler = self._build_scalers(train_dataset)

        train_loader = self._loader(
            ScaledForecastingDataset(train_dataset, scaler),
            shuffle=True,
            seed_offset=0,
        )
        val_loader = None
        if len(val_dataset) > 0:
            val_loader = self._loader(
                ScaledForecastingDataset(val_dataset, scaler),
                shuffle=False,
                seed_offset=1,
            )
        test_loader = self._loader(
            ScaledForecastingDataset(test_dataset, scaler),
            shuffle=False,
            seed_offset=2,
        )

        if resume_checkpoint is not None:
            model = restore_model_from_checkpoint(resume_checkpoint, self.config.model).to(device)
        else:
            model = build_model(
                self.config.model,
                input_len=self.config.data.input_len,
                output_len=self.config.data.output_len,
                input_dim=train_dataset.input_dim,
                target_dim=train_dataset.target_dim,
            ).to(device)
        optimizer = self._build_optimizer(model)
        scheduler = self._build_scheduler(optimizer)
        start_epoch = 1
        resumed_from = self.config.training.resume_from
        if resume_checkpoint is not None:
            load_optimizer_state_from_checkpoint(resume_checkpoint, optimizer)
            start_epoch = int(resume_checkpoint["epoch"]) + 1
            logger.info("Resuming from %s at epoch %s", resumed_from, start_epoch)

        history: list[dict[str, Any]] = []
        checkpoint_path = run_dir / "checkpoint.pt"
        best_checkpoint_path = run_dir / "best_checkpoint.pt"
        best_epoch: int | None = None
        best_metric: dict[str, Any] | None = None
        best_metric_value: float | None = None
        best_model_state: dict[str, torch.Tensor] | None = None
        early_stop_counter = 0
        early_stopped = False
        last_epoch = start_epoch - 1
        validation_metrics: OptionalMetricGroups = None
        training_started_at = time.monotonic()
        target_duration_seconds = self._target_duration_seconds()
        target_epoch_seconds = self._target_epoch_seconds()
        self._write_progress(
            run_dir,
            recorder=recorder,
            status="running",
            history=history,
            latest=None,
            elapsed_seconds=0.0,
            target_duration_seconds=target_duration_seconds,
            target_epoch_seconds=target_epoch_seconds,
        )

        if start_epoch > self.config.training.epochs:
            logger.info(
                "Checkpoint epoch is already >= target epochs; "
                "skipping training and evaluating only."
            )

        for epoch in range(start_epoch, self.config.training.epochs + 1):
            last_epoch = epoch
            epoch_started_at = time.monotonic()
            train_loss = self._train_one_epoch(model, train_loader, optimizer, device)
            if val_loader is not None:
                validation_metrics = evaluate(
                    model,
                    val_loader,
                    self.config.evaluation.metrics,
                    device,
                    scaler=_target_scaler(scaler),
                    include_scaled_metrics=self.config.evaluation.include_scaled_metrics,
                )
            metric_name = self._best_metric_name()
            metric_value = _metric_value(validation_metrics, metric_name)
            improved = _is_improved(
                metric_value,
                best_metric_value,
                mode=self.config.training.early_stopping.mode,
                min_delta=self.config.training.early_stopping.min_delta,
            )
            if improved and metric_value is not None:
                best_metric_value = metric_value
                best_epoch = epoch
                best_metric = {
                    "name": metric_name,
                    "value": metric_value,
                    "mode": self.config.training.early_stopping.mode,
                }
                best_model_state = _clone_model_state(model)
                save_checkpoint(
                    best_checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    metrics={
                        "validation_metrics": validation_metrics,
                        "train_loss": train_loss,
                        "best_metric": best_metric,
                    },
                    config=self.config,
                    scaler=scaler,
                    environment=environment,
                )
                early_stop_counter = 0
            elif metric_value is not None:
                early_stop_counter += 1
            row: dict[str, Any] = {"epoch": epoch, "train_loss": train_loss}
            if validation_metrics is not None:
                row["validation_metrics"] = validation_metrics
            row["learning_rate"] = _current_learning_rate(optimizer)
            row["best_epoch"] = best_epoch
            row["best_metric"] = best_metric
            row["early_stop_counter"] = early_stop_counter
            row["early_stopped"] = False
            history.append(row)
            logger.info("epoch=%s train_loss=%.6f val=%s", epoch, train_loss, validation_metrics)
            self._write_progress(
                run_dir,
                recorder=recorder,
                status="running",
                history=history,
                latest=row,
                best_epoch=best_epoch,
                best_metric=best_metric,
                early_stopped=early_stopped,
                elapsed_seconds=time.monotonic() - training_started_at,
                target_duration_seconds=target_duration_seconds,
                target_epoch_seconds=target_epoch_seconds,
            )
            self._pace_epoch(
                run_dir,
                recorder=recorder,
                epoch_started_at=epoch_started_at,
                training_started_at=training_started_at,
                history=history,
                latest=row,
                best_epoch=best_epoch,
                best_metric=best_metric,
                early_stopped=early_stopped,
                target_duration_seconds=target_duration_seconds,
                target_epoch_seconds=target_epoch_seconds,
            )

            should_checkpoint = (
                self.config.training.checkpoint_every
                and epoch % self.config.training.checkpoint_every == 0
            )
            if should_checkpoint:
                checkpoint_path = save_checkpoint(
                    run_dir / f"checkpoint_epoch_{epoch}.pt",
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    metrics={"validation_metrics": validation_metrics, "train_loss": train_loss},
                    config=self.config,
                    scaler=scaler,
                    environment=environment,
                )
            if scheduler is not None:
                scheduler.step()
            if self._should_stop_early(metric_value, early_stop_counter):
                row["early_stopped"] = True
                early_stopped = True
                logger.info("early stopping at epoch=%s best_epoch=%s", epoch, best_epoch)
                self._write_progress(
                    run_dir,
                    recorder=recorder,
                    status="running",
                    history=history,
                    latest=row,
                    best_epoch=best_epoch,
                    best_metric=best_metric,
                    early_stopped=early_stopped,
                    elapsed_seconds=time.monotonic() - training_started_at,
                    target_duration_seconds=target_duration_seconds,
                    target_epoch_seconds=target_epoch_seconds,
                )
                break

        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            final_epoch = best_epoch or last_epoch
        else:
            final_epoch = last_epoch

        test_metrics = evaluate(
            model,
            test_loader,
            self.config.evaluation.metrics,
            device,
            scaler=_target_scaler(scaler),
            include_scaled_metrics=self.config.evaluation.include_scaled_metrics,
        )
        forecast_samples = collect_forecast_samples(
            model,
            test_loader,
            device,
            scaler=_target_scaler(scaler),
            target_cols=_dataset_columns(train_dataset, "target_cols"),
        )
        checkpoint_path = save_checkpoint(
            run_dir / "checkpoint.pt",
            model=model,
            optimizer=optimizer,
            epoch=final_epoch,
            metrics={"validation_metrics": validation_metrics, "test_metrics": test_metrics},
            config=self.config,
            scaler=scaler,
            environment=environment,
        )
        data_metadata = _data_metadata(train_dataset)
        model_export_path, model_export_metadata_path = save_model_export(
            run_dir / "model_export.pt",
            run_dir / "model_export.json",
            model=model,
            config=self.config,
            scaler=scaler,
            metrics={"validation_metrics": validation_metrics, "test_metrics": test_metrics},
            data_metadata=data_metadata,
        )
        metadata = recorder.metadata()
        result = TrainingResult(
            run_dir=run_dir,
            run_id=metadata["run_id"],
            created_at=metadata["created_at"],
            experiment_name=metadata["experiment_name"],
            checkpoint_path=checkpoint_path,
            best_checkpoint_path=(
                best_checkpoint_path if best_checkpoint_path.exists() else checkpoint_path
            ),
            best_epoch=best_epoch,
            best_metric=best_metric,
            model_export_path=model_export_path,
            model_export_metadata_path=model_export_metadata_path,
            history=history,
            validation_metrics=validation_metrics,
            test_metrics=test_metrics,
            data_metadata=data_metadata,
            forecast_samples=forecast_samples,
            resumed_from=resumed_from,
        )
        logger.info("test=%s", test_metrics)
        self._write_progress(
            run_dir,
            recorder=recorder,
            status="succeeded",
            history=history,
            latest=history[-1] if history else None,
            test_metrics=test_metrics,
            best_epoch=best_epoch,
            best_metric=best_metric,
            early_stopped=early_stopped,
            elapsed_seconds=time.monotonic() - training_started_at,
            target_duration_seconds=target_duration_seconds,
            target_epoch_seconds=target_epoch_seconds,
        )
        recorder.save_results(result.to_dict())
        (run_dir / "forecast_samples.json").write_text(
            json.dumps(forecast_samples, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        save_artifact_manifest(
            build_train_artifact_manifest(
                experiment_name=result.experiment_name,
                run_id=result.run_id,
                run_dir=result.run_dir,
                checkpoint_path=result.checkpoint_path,
                model_export_path=result.model_export_path,
                model_export_metadata_path=result.model_export_metadata_path,
                best_checkpoint_path=result.best_checkpoint_path,
            ),
            run_dir / "artifacts.json",
        )
        return result

    def _write_progress(
        self,
        run_dir: Path,
        *,
        recorder: ExperimentRecorder,
        status: str,
        history: list[dict[str, Any]],
        latest: dict[str, Any] | None,
        test_metrics: MetricGroups | None = None,
        best_epoch: int | None = None,
        best_metric: dict[str, Any] | None = None,
        early_stopped: bool = False,
        elapsed_seconds: float | None = None,
        target_duration_seconds: float | None = None,
        target_epoch_seconds: float | None = None,
        pacing_state: dict[str, Any] | None = None,
    ) -> None:
        total_epochs = self.config.training.epochs
        completed_epochs = int(latest["epoch"]) if latest and "epoch" in latest else 0
        payload: dict[str, Any] = {
            "status": status,
            "run_id": recorder.run_id,
            "experiment_name": self.config.experiment.name,
            "run_dir": str(run_dir),
            "updated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "total_epochs": total_epochs,
            "completed_epochs": completed_epochs,
            "progress_percent": round((completed_epochs / total_epochs) * 100, 2),
            "latest": latest,
            "history": history,
            "best_epoch": best_epoch,
            "best_metric": best_metric,
            "early_stopped": early_stopped,
        }
        if elapsed_seconds is not None:
            payload["elapsed_seconds"] = round(max(0.0, elapsed_seconds), 2)
        if target_duration_seconds is not None:
            payload["target_duration_minutes"] = self.config.training.target_duration_minutes
            payload["target_duration_seconds"] = round(target_duration_seconds, 2)
            payload["estimated_remaining_seconds"] = round(
                max(0.0, target_duration_seconds - (elapsed_seconds or 0.0)),
                2,
            )
        if target_epoch_seconds is not None:
            payload["target_epoch_seconds"] = round(target_epoch_seconds, 2)
        if pacing_state is not None:
            payload["pacing"] = pacing_state
        if test_metrics is not None:
            payload["test_metrics"] = test_metrics
        (run_dir / "progress.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _target_duration_seconds(self) -> float | None:
        if self.config.training.target_duration_minutes is None:
            return None
        return self.config.training.target_duration_minutes * 60.0

    def _target_epoch_seconds(self) -> float | None:
        target_duration_seconds = self._target_duration_seconds()
        if target_duration_seconds is None:
            return None
        return target_duration_seconds / self.config.training.epochs

    def _pace_epoch(
        self,
        run_dir: Path,
        *,
        recorder: ExperimentRecorder,
        epoch_started_at: float,
        training_started_at: float,
        history: list[dict[str, Any]],
        latest: dict[str, Any],
        best_epoch: int | None,
        best_metric: dict[str, Any] | None,
        early_stopped: bool,
        target_duration_seconds: float | None,
        target_epoch_seconds: float | None,
    ) -> None:
        if target_epoch_seconds is None or target_duration_seconds is None:
            return
        while True:
            epoch_elapsed = time.monotonic() - epoch_started_at
            remaining = target_epoch_seconds - epoch_elapsed
            if remaining <= 0:
                return
            sleep_seconds = min(remaining, 3.0)
            time.sleep(sleep_seconds)
            epoch_elapsed = time.monotonic() - epoch_started_at
            self._write_progress(
                run_dir,
                recorder=recorder,
                status="running",
                history=history,
                latest=latest,
                best_epoch=best_epoch,
                best_metric=best_metric,
                early_stopped=early_stopped,
                elapsed_seconds=time.monotonic() - training_started_at,
                target_duration_seconds=target_duration_seconds,
                target_epoch_seconds=target_epoch_seconds,
                pacing_state={
                    "active": True,
                    "epoch_elapsed_seconds": round(max(0.0, epoch_elapsed), 2),
                    "epoch_sleep_remaining_seconds": round(max(0.0, remaining - sleep_seconds), 2),
                },
            )

    def _loader(
        self,
        dataset: ScaledForecastingDataset,
        *,
        shuffle: bool,
        seed_offset: int,
    ) -> DataLoader[ForecastBatch]:
        seed = self.config.experiment.seed + seed_offset
        generator = torch.Generator().manual_seed(seed)
        return DataLoader(
            dataset,
            batch_size=self.config.data.batch_size,
            shuffle=shuffle,
            num_workers=self.config.training.num_workers,
            generator=generator,
            worker_init_fn=build_worker_init_fn(seed),
        )

    def _build_scalers(self, train_dataset: ForecastingDataset) -> ScalerOrBundle:
        """Build fitted target-only or feature-aware scalers for a training dataset."""

        target_scaler = build_scaler(self.config.data.scaler)
        target_scaler.fit(train_dataset.target_scaler_fit_values())
        if train_dataset.input_dim == train_dataset.target_dim:
            return target_scaler

        feature_scaler = build_scaler(self.config.data.scaler)
        feature_scaler.fit(train_dataset.feature_scaler_fit_values())
        return FeatureAwareScalerBundle(target=target_scaler, features=feature_scaler)

    def _resolve_device(self) -> torch.device:
        return resolve_training_device(self.config.training.device)

    def _build_optimizer(self, model: nn.Module) -> torch.optim.Optimizer | None:
        params = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not params:
            return None
        if self.config.training.optimizer == "adam":
            return torch.optim.Adam(params, lr=self.config.training.learning_rate)
        if self.config.training.optimizer == "sgd":
            return torch.optim.SGD(params, lr=self.config.training.learning_rate)
        msg = f"unsupported optimizer: {self.config.training.optimizer}"
        raise ValueError(msg)

    def _build_scheduler(
        self,
        optimizer: torch.optim.Optimizer | None,
    ) -> torch.optim.lr_scheduler.LRScheduler | None:
        if optimizer is None:
            return None
        scheduler_config = self.config.training.lr_scheduler
        if scheduler_config.name == "none":
            return None
        if scheduler_config.name == "step":
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=scheduler_config.step_size,
                gamma=scheduler_config.gamma,
            )
        if scheduler_config.name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config.training.epochs,
                eta_min=scheduler_config.eta_min,
            )
        msg = f"unsupported lr_scheduler: {scheduler_config.name}"
        raise ValueError(msg)

    def _best_metric_name(self) -> str:
        return (
            self.config.training.early_stopping.metric
            or self.config.training.best_checkpoint_metric
        )

    def _should_stop_early(self, metric_value: float | None, early_stop_counter: int) -> bool:
        if not self.config.training.early_stopping.enabled:
            return False
        if metric_value is None:
            return False
        return early_stop_counter >= self.config.training.early_stopping.patience

    def _train_one_epoch(
        self,
        model: nn.Module,
        batches: DataLoader[ForecastBatch],
        optimizer: torch.optim.Optimizer | None,
        device: torch.device,
    ) -> float:
        model.train()
        losses: list[float] = []
        for batch in batches:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            y_pred = model(x)
            loss = self._loss(y_pred, y)
            if optimizer is not None:
                loss.backward()  # type: ignore[no-untyped-call]
                if self.config.training.gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        self.config.training.gradient_clip_norm,
                    )
                optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        if not losses:
            msg = "cannot train with no batches"
            raise ValueError(msg)
        return sum(losses) / len(losses)

    def _loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if self.config.training.loss == "mse":
            return torch.mean((y_pred - y_true) ** 2)
        if self.config.training.loss == "mae":
            return torch.mean(torch.abs(y_pred - y_true))
        msg = f"unsupported loss: {self.config.training.loss}"
        raise ValueError(msg)

    def _load_resume_checkpoint(self) -> CheckpointPayload | None:
        if self.config.training.resume_from is None:
            return None
        return load_checkpoint(self.config.training.resume_from)


def _target_scaler(scaler: ScalerOrBundle) -> BaseScaler:
    if isinstance(scaler, FeatureAwareScalerBundle):
        return scaler.target
    return scaler


def _dataset_columns(dataset: ForecastingDataset, name: str) -> list[str]:
    value = getattr(dataset, name, [])
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _data_metadata(dataset: ForecastingDataset) -> dict[str, Any]:
    return {
        "input_dim": dataset.input_dim,
        "target_dim": dataset.target_dim,
        "feature_dim": dataset.feature_dim,
        "target_cols": _dataset_columns(dataset, "target_cols"),
        "feature_cols": _dataset_columns(dataset, "feature_cols"),
        "feature_aware": dataset.input_dim != dataset.target_dim,
    }


def _metric_value(metrics: OptionalMetricGroups, metric_name: str) -> float | None:
    if metrics is None:
        return None
    original = metrics.get("original")
    if not isinstance(original, dict):
        return None
    value = original.get(metric_name)
    if value is None:
        return None
    return float(value)


def _is_improved(
    metric_value: float | None,
    best_metric_value: float | None,
    *,
    mode: str,
    min_delta: float,
) -> bool:
    if metric_value is None:
        return False
    if best_metric_value is None:
        return True
    if mode == "max":
        return metric_value > best_metric_value + min_delta
    return metric_value < best_metric_value - min_delta


def _clone_model_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _current_learning_rate(optimizer: torch.optim.Optimizer | None) -> float | None:
    if optimizer is None or not optimizer.param_groups:
        return None
    return float(optimizer.param_groups[0]["lr"])


def run_training(config_path: str | Path, logger: logging.Logger | None = None) -> TrainingResult:
    """Convenience function for scripts and API callers."""

    result = Trainer.from_config_path(config_path).run()
    if logger is not None:
        logger.info("training completed: %s", result.to_dict())
    return result
