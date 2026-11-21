"""Ridge linear regression solved exactly from the normal equations.

Mathematical contract
---------------------
Given features X (n rows, p columns) and targets y (length n), fit the affine
predictor y_hat = X w + b by minimizing

    L(w, b) = ||y - X w - b||^2 + alpha * ||w||^2,      alpha >= 0.

The intercept b is not penalized. Centering removes it: with training means
x_bar and y_bar, Xc = X - x_bar and yc = y - y_bar, the minimizer solves

    (Xc^T Xc + alpha I) w = Xc^T yc,      b = y_bar - x_bar . w.

For alpha > 0 the matrix is positive definite, so the solution is unique even
when columns are collinear (for example, first- and second-floor area summing
to living area). At alpha = 0 this is ordinary least squares, solved with a
least-squares routine that tolerates rank deficiency.

When ``standardize`` is True, columns are scaled to unit variance before
fitting, so one alpha penalizes every feature on the same scale. Coefficients
are reported on the original feature scale.
"""

from __future__ import annotations

from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

from dsm_project.algorithms.student._validation import (
    Standardizer,
    check_features,
    check_fitted,
    check_prediction_features,
    check_targets,
)


class RidgeRegression:
    """Closed-form L2-penalized least squares with an unpenalized intercept.

    Parameters
    ----------
    alpha
        Nonnegative penalty strength. ``0`` gives ordinary least squares.
    standardize
        Scale features to unit variance (training statistics) before fitting.

    Attributes
    ----------
    coef_
        Weights on the original feature scale, shape ``(p,)``.
    intercept_
        Intercept on the original scale.
    n_features_in_
        Number of features seen during ``fit``.
    """

    def __init__(self, *, alpha: float = 1.0, standardize: bool = True) -> None:
        if not np.isfinite(alpha) or alpha < 0:
            raise ValueError("alpha must be a finite nonnegative number")
        self.alpha = float(alpha)
        self.standardize = bool(standardize)

    def fit(self, features: ArrayLike, targets: ArrayLike) -> Self:
        """Solve the penalized normal equations for the training data."""

        x = check_features(features)
        y = check_targets(targets, x.shape[0])
        scaler = Standardizer().fit(x) if self.standardize else None
        z = scaler.transform(x) if scaler else x

        z_mean = z.mean(axis=0)
        y_mean = float(y.mean())
        z_centered = z - z_mean
        y_centered = y - y_mean
        if self.alpha > 0:
            gram = z_centered.T @ z_centered + self.alpha * np.eye(z.shape[1])
            weights = np.linalg.solve(gram, z_centered.T @ y_centered)
        else:
            weights = np.linalg.lstsq(z_centered, y_centered, rcond=None)[0]
        intercept = y_mean - float(z_mean @ weights)

        if scaler:
            # Map z = (x - m) / s back to x: w_x = w_z / s, b_x = b_z - m . w_x.
            weights = weights / scaler.scale_
            intercept = intercept - float(scaler.mean_ @ weights)
        self.coef_ = weights
        self.intercept_ = intercept
        self.n_features_in_ = x.shape[1]
        return self

    def predict(self, features: ArrayLike) -> NDArray[np.float64]:
        """Return X w + b for each row."""

        check_fitted(self, "coef_")
        x = check_prediction_features(features, self.n_features_in_)
        return x @ self.coef_ + self.intercept_


def ridge_objective(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    weights: NDArray[np.float64],
    intercept: float,
    alpha: float,
) -> float:
    """Return the ridge objective ||y - X w - b||^2 + alpha * ||w||^2."""

    residual = y - x @ weights - intercept
    return float(residual @ residual + alpha * weights @ weights)


def ridge_gradient(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    weights: NDArray[np.float64],
    intercept: float,
    alpha: float,
) -> tuple[NDArray[np.float64], float]:
    """Return the gradient (dL/dw, dL/db) of the ridge objective.

    With residual r = y - X w - b: dL/dw = -2 X^T r + 2 alpha w and
    dL/db = -2 sum(r). Both vanish at the fitted solution.
    """

    residual = y - x @ weights - intercept
    return -2.0 * x.T @ residual + 2.0 * alpha * weights, float(-2.0 * residual.sum())
