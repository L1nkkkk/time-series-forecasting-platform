"""Training runner implementation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from ts_platform.config.loader import load_config
from ts_platform.config.schema import PlatformConfig
from ts_platform.data.base import ForecastBatch
from ts_platform.data.loaders import build_dataset
from ts_platform.data.transforms import ScaledForecastingDataset
from ts_platform.experiment.artifacts import build_train_artifact_manifest, save_artifact_manifest
from ts_platform.experiment.logger import setup_experiment_logger
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
    load_checkpoint,
    load_optimizer_state_from_checkpoint,
    restore_model_from_checkpoint,
    restore_scaler_from_checkpoint,
    save_checkpoint,
    validate_checkpoint_for_training,
)
from ts_platform.runner.evaluator import evaluate
from ts_platform.scaler.registry import build_scaler


@dataclass(frozen=True)
class TrainingResult:
    """Serializable training result summary."""

    run_dir: Path
    run_id: str
    created_at: str
    experiment_name: str
    checkpoint_path: Path
    history: list[dict[str, Any]]
    validation_metrics: OptionalMetricGroups
    test_metrics: MetricGroups
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
            history=self.history,
            validation_metrics=self.validation_metrics,
            test_metrics=self.test_metrics,
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
        resume_checkpoint = self._load_resume_checkpoint()
        recorder = ExperimentRecorder(
            self.config.experiment.output_dir,
            self.config.experiment.name,
            overwrite=self.config.experiment.overwrite,
        )
        run_dir = recorder.prepare()
        logger = setup_experiment_logger(run_dir)
        recorder.save_config(self.config)
        environment = collect_environment()
        recorder.save_environment(environment)

        device = self._resolve_device()
        logger.info("Using device: %s", device)

        train_dataset = build_dataset(self.config.data, "train", self.config.experiment.seed)
        val_dataset = build_dataset(self.config.data, "val", self.config.experiment.seed)
        test_dataset = build_dataset(self.config.data, "test", self.config.experiment.seed)
        if train_dataset.input_dim != train_dataset.target_dim:
            msg = "feature-aware training is not implemented until Phase 12D/12E"
            raise NotImplementedError(msg)

        if resume_checkpoint is not None:
            validate_checkpoint_for_training(
                resume_checkpoint,
                self.config,
                num_features=train_dataset.num_features,
            )
            scaler = restore_scaler_from_checkpoint(resume_checkpoint, self.config.data.scaler)
        else:
            scaler = build_scaler(self.config.data.scaler)
            scaler.fit(train_dataset.scaler_fit_values())

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
                num_features=train_dataset.num_features,
            ).to(device)
        optimizer = self._build_optimizer(model)
        start_epoch = 1
        resumed_from = self.config.training.resume_from
        if resume_checkpoint is not None:
            load_optimizer_state_from_checkpoint(resume_checkpoint, optimizer)
            start_epoch = int(resume_checkpoint["epoch"]) + 1
            logger.info("Resuming from %s at epoch %s", resumed_from, start_epoch)

        history: list[dict[str, Any]] = []
        checkpoint_path = run_dir / "checkpoint.pt"
        validation_metrics: OptionalMetricGroups = None

        if start_epoch > self.config.training.epochs:
            logger.info(
                "Checkpoint epoch is already >= target epochs; "
                "skipping training and evaluating only."
            )

        for epoch in range(start_epoch, self.config.training.epochs + 1):
            train_loss = self._train_one_epoch(model, train_loader, optimizer, device)
            if val_loader is not None:
                validation_metrics = evaluate(
                    model,
                    val_loader,
                    self.config.evaluation.metrics,
                    device,
                    scaler=scaler,
                    include_scaled_metrics=self.config.evaluation.include_scaled_metrics,
                )
            row: dict[str, Any] = {"epoch": epoch, "train_loss": train_loss}
            if validation_metrics is not None:
                row["validation_metrics"] = validation_metrics
            history.append(row)
            logger.info("epoch=%s train_loss=%.6f val=%s", epoch, train_loss, validation_metrics)

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

        test_metrics = evaluate(
            model,
            test_loader,
            self.config.evaluation.metrics,
            device,
            scaler=scaler,
            include_scaled_metrics=self.config.evaluation.include_scaled_metrics,
        )
        final_epoch = max(start_epoch - 1, self.config.training.epochs)
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
        metadata = recorder.metadata()
        result = TrainingResult(
            run_dir=run_dir,
            run_id=metadata["run_id"],
            created_at=metadata["created_at"],
            experiment_name=metadata["experiment_name"],
            checkpoint_path=checkpoint_path,
            history=history,
            validation_metrics=validation_metrics,
            test_metrics=test_metrics,
            resumed_from=resumed_from,
        )
        logger.info("test=%s", test_metrics)
        recorder.save_results(result.to_dict())
        save_artifact_manifest(
            build_train_artifact_manifest(
                experiment_name=result.experiment_name,
                run_id=result.run_id,
                run_dir=result.run_dir,
                checkpoint_path=result.checkpoint_path,
            ),
            run_dir / "artifacts.json",
        )
        return result

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

    def _resolve_device(self) -> torch.device:
        device = torch.device(self.config.training.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            msg = "CUDA was requested but is not available"
            raise RuntimeError(msg)
        return device

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


def run_training(config_path: str | Path, logger: logging.Logger | None = None) -> TrainingResult:
    """Convenience function for scripts and API callers."""

    result = Trainer.from_config_path(config_path).run()
    if logger is not None:
        logger.info("training completed: %s", result.to_dict())
    return result
