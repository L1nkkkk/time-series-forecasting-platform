from __future__ import annotations

import math

import pytest
import torch

from ts_platform.metrics.regression import compute_metrics, mae, mape, mse, rmse, wape


def test_regression_metrics() -> None:
    y_true = torch.tensor([1.0, 2.0, 4.0])
    y_pred = torch.tensor([1.0, 3.0, 2.0])

    assert mae(y_pred, y_true) == pytest.approx(1.0)
    assert mse(y_pred, y_true) == pytest.approx(5.0 / 3.0)
    assert rmse(y_pred, y_true) == pytest.approx(math.sqrt(5.0 / 3.0))
    assert mape(y_pred, y_true) == pytest.approx((0.0 + 0.5 + 0.5) / 3.0)
    assert wape(y_pred, y_true) == pytest.approx(3.0 / 7.0)


def test_metrics_handle_zero_targets() -> None:
    y_true = torch.zeros(2)
    y_pred = torch.ones(2)

    results = compute_metrics(y_pred, y_true, ["mape", "wape"])

    assert math.isfinite(results["mape"])
    assert math.isfinite(results["wape"])


def test_metrics_reject_empty_and_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="empty"):
        mae(torch.tensor([]), torch.tensor([]))

    with pytest.raises(ValueError, match="shape mismatch"):
        mae(torch.zeros(2), torch.zeros(2, 1))
