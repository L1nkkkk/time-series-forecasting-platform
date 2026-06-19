from __future__ import annotations

import torch

from ts_platform.scaler.base import IdentityScaler
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


def test_standard_scaler_state_roundtrip() -> None:
    values = torch.tensor([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]])
    scaler = StandardScaler().fit(values)
    restored = StandardScaler()

    restored.load_state_dict(scaler.state_dict())

    assert torch.allclose(restored.transform(values), scaler.transform(values))
    assert torch.allclose(restored.inverse_transform(scaler.transform(values)), values)


def test_minmax_scaler_state_roundtrip() -> None:
    values = torch.tensor([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]])
    scaler = MinMaxScaler(feature_range=(-1.0, 1.0)).fit(values)
    restored = MinMaxScaler()

    restored.load_state_dict(scaler.state_dict())

    assert torch.allclose(restored.transform(values), scaler.transform(values))
    assert torch.allclose(restored.inverse_transform(scaler.transform(values)), values)


def test_identity_scaler_state_roundtrip() -> None:
    values = torch.tensor([[1.0], [2.0]])
    scaler = IdentityScaler().fit(values)
    restored = IdentityScaler()

    restored.load_state_dict(scaler.state_dict())

    assert torch.equal(restored.transform(values), values)
    assert torch.equal(restored.inverse_transform(values), values)
