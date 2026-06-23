"""Torch device availability helpers."""

from __future__ import annotations

from typing import Any

import torch


class DeviceUnavailableError(RuntimeError):
    """Raised when a requested torch device cannot be used by this process."""


def resolve_training_device(device_name: str) -> torch.device:
    """Return a torch device after validating runtime availability."""

    device = torch.device(device_name)
    if device.type == "cuda":
        _validate_cuda_device(device)
    return device


def device_status() -> dict[str, Any]:
    """Return runtime device availability for API clients."""

    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0
    return {
        "cuda_available": cuda_available,
        "cuda_device_count": device_count,
    }


def _validate_cuda_device(device: torch.device) -> None:
    if not torch.cuda.is_available():
        msg = "CUDA was requested but is not available on this machine"
        raise DeviceUnavailableError(msg)
    device_count = torch.cuda.device_count()
    if device.index is not None and device.index >= device_count:
        msg = (
            f"CUDA device index {device.index} was requested, "
            f"but only {device_count} CUDA device(s) are available"
        )
        raise DeviceUnavailableError(msg)
