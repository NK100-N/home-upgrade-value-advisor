"""Leakage-resistant partitions: a sealed time-based final test and CV folds."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from dsm_project.config import ProjectConfig


@dataclass(frozen=True)
class TimePartition:
    """Development rows for fitting and selection; final-test rows sealed."""

    development: pd.DataFrame
    final_test: pd.DataFrame


def time_partition(observations: pd.DataFrame, config: ProjectConfig) -> TimePartition:
    """Split rows by the configured time column and final-test values.

    Raises
    ------
    ValueError
        If the split is not configured or either side would be empty.
    """

    time_column = config.split.get("time_column")
    final_values = config.split.get("final_test_values")
    if not time_column or not final_values:
        raise ValueError("split.time_column and split.final_test_values are required")
    if time_column not in observations:
        raise ValueError(f"Missing time column: {time_column}")
    is_final = observations[time_column].isin([int(v) for v in final_values])
    development = observations.loc[~is_final]
    final_test = observations.loc[is_final]
    if development.empty or final_test.empty:
        raise ValueError("Both development and final-test partitions must be nonempty")
    return TimePartition(development=development.copy(), final_test=final_test.copy())


def has_sealed_final_test(config: ProjectConfig) -> bool:
    """Return True when the configuration declares a sealed final test."""

    return bool(
        config.split.get("time_column") and config.split.get("final_test_values")
    )


def kfold_indices(
    row_count: int, folds: int, seed: int
) -> list[tuple[NDArray[np.int64], NDArray[np.int64]]]:
    """Return reproducible shuffled K-fold (train, validation) index pairs."""

    if folds < 2:
        raise ValueError("folds must be at least 2")
    if row_count < folds:
        raise ValueError("Need at least one row per fold")
    order = np.random.default_rng(seed).permutation(row_count)
    pieces = np.array_split(order, folds)
    return [
        (np.sort(np.concatenate(pieces[:i] + pieces[i + 1 :])), np.sort(piece))
        for i, piece in enumerate(pieces)
    ]
