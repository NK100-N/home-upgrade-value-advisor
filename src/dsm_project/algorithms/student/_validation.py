"""Input checks shared by every student estimator."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


def check_features(features: ArrayLike) -> NDArray[np.float64]:
    """Return a finite, nonempty, two-dimensional float array."""

    array = np.asarray(features, dtype=float)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("features must be a nonempty two-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError("features must contain only finite values")
    return array


def check_targets(targets: ArrayLike, rows: int) -> NDArray[np.float64]:
    """Return a finite target vector aligned with ``rows`` observations."""

    array = np.asarray(targets, dtype=float)
    if array.shape != (rows,):
        raise ValueError("targets must contain one value per observation")
    if not np.isfinite(array).all():
        raise ValueError("targets must contain only finite values")
    return array


def check_binary_labels(labels: ArrayLike, rows: int) -> NDArray[np.int64]:
    """Return a 0/1 label vector aligned with ``rows`` observations."""

    array = np.asarray(labels)
    if array.shape != (rows,):
        raise ValueError("labels must contain one value per observation")
    values = set(np.unique(array).tolist())
    if not values <= {0, 1}:
        raise ValueError("labels must be binary 0/1")
    return array.astype(np.int64)


def check_fitted(estimator: Any, attribute: str) -> None:
    """Raise if ``estimator`` has not been fitted."""

    if not hasattr(estimator, attribute):
        raise RuntimeError("fit must be called before using this estimator")


def check_prediction_features(
    features: ArrayLike, expected: int
) -> NDArray[np.float64]:
    """Validate prediction rows against the fitted feature count."""

    array = check_features(features)
    if array.shape[1] != expected:
        raise ValueError(f"expected {expected} features, received {array.shape[1]}")
    return array


class Standardizer:
    """Center and scale columns using training statistics only.

    Constant columns keep a unit scale so that they map to zero, not NaN.
    """

    def fit(self, features: NDArray[np.float64]) -> Standardizer:
        self.mean_ = features.mean(axis=0)
        scale = features.std(axis=0)
        self.scale_ = np.where(scale > 0, scale, 1.0)
        return self

    def transform(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        return (features - self.mean_) / self.scale_
