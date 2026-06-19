"""Dataset split helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SplitIndices:
    """Integer indices for train, validation, and test splits."""

    train: list[int]
    val: list[int]
    test: list[int]


@dataclass(frozen=True)
class TimeSplitSlices:
    """Raw-row slices for train, validation, and test splits."""

    train: slice
    val: slice
    test: slice

    def for_mode(self, mode: str) -> slice:
        """Return the slice for a split name."""

        if mode == "train":
            return self.train
        if mode == "val":
            return self.val
        if mode == "test":
            return self.test
        msg = f"unknown split mode: {mode}"
        raise ValueError(msg)


def compute_split_indices(
    total: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> SplitIndices:
    """Split integer positions into train/validation/test lists."""

    if total <= 0:
        msg = "total must be positive"
        raise ValueError(msg)
    ratio_total = train_ratio + val_ratio + test_ratio
    if abs(ratio_total - 1.0) > 1e-6:
        msg = "split ratios must sum to 1.0"
        raise ValueError(msg)

    train_end = max(1, int(total * train_ratio))
    val_count = int(total * val_ratio)
    val_end = min(total, train_end + val_count)
    if val_ratio > 0 and val_end == train_end and total - train_end > 1:
        val_end += 1
    if val_end >= total:
        val_end = total - 1
    train = list(range(0, train_end))
    val = list(range(train_end, val_end))
    test = list(range(val_end, total))
    if not test:
        msg = "split configuration leaves no test samples"
        raise ValueError(msg)
    return SplitIndices(train=train, val=val, test=test)


def compute_time_split_slices(
    total: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    *,
    min_length: int,
) -> TimeSplitSlices:
    """Split raw row positions before generating sliding windows."""

    if total <= 0:
        msg = "total must be positive"
        raise ValueError(msg)
    if min_length <= 0:
        msg = "min_length must be positive"
        raise ValueError(msg)
    ratio_total = train_ratio + val_ratio + test_ratio
    if abs(ratio_total - 1.0) > 1e-6:
        msg = "split ratios must sum to 1.0"
        raise ValueError(msg)

    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    slices = TimeSplitSlices(
        train=slice(0, train_end),
        val=slice(train_end, val_end),
        test=slice(val_end, total),
    )
    for name, split_slice, ratio in [
        ("train", slices.train, train_ratio),
        ("val", slices.val, val_ratio),
        ("test", slices.test, test_ratio),
    ]:
        length = split_slice.stop - split_slice.start
        if ratio > 0 and length < min_length:
            msg = (
                f"{name} split requires at least {min_length} rows to create one window; "
                f"got {length}"
            )
            raise ValueError(msg)
    return slices
