from __future__ import annotations

import torch

from ts_platform.scaler.minmax import MinMaxScaler
from ts_platform.scaler.standard import StandardScaler


def test_standard_scaler_roundtrip() -> None:
    values = torch.tensor([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]])
    scaler = StandardScaler().fit(values)

    transformed = scaler.transform(values)
    restored = scaler.inverse_transform(transformed)

    assert torch.allclose(transformed.mean(dim=0), torch.zeros(2), atol=1e-6)
    assert torch.allclose(restored, values, atol=1e-6)


def test_minmax_scaler_roundtrip() -> None:
    values = torch.tensor([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]])
    scaler = MinMaxScaler().fit(values)

    transformed = scaler.transform(values)
    restored = scaler.inverse_transform(transformed)

    assert torch.all(transformed >= 0)
    assert torch.all(transformed <= 1)
    assert torch.allclose(restored, values, atol=1e-6)
