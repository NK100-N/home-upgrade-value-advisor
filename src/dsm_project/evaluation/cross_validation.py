"""Development-only cross-validation and the baselines every model must beat.

Nothing in this module reads the sealed final-test partition.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    mean_absolute_error,
    root_mean_squared_error,
)

from dsm_project.config import ProjectConfig
from dsm_project.contracts import ProjectDataset
from dsm_project.data.splits import kfold_indices, time_partition

FoldPredictor = Callable[
    [pd.DataFrame, pd.DataFrame], NDArray[np.float64] | NDArray[np.int64]
]


def _folds(config: ProjectConfig, rows: int) -> list[tuple[Any, Any]]:
    folds = int(config.split.get("cv_folds", 5))
    seed = int(config.split.get("random_state", 438))
    return kfold_indices(rows, folds, seed)


def regression_scores(observed: Any, predicted: Any) -> dict[str, float]:
    """Return dollar-scale and log-scale regression errors."""

    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    scores = {
        "rmse": float(root_mean_squared_error(observed, predicted)),
        "mae": float(mean_absolute_error(observed, predicted)),
    }
    if (predicted > 0).all():
        scores["rmse_log"] = float(
            root_mean_squared_error(np.log(observed), np.log(predicted))
        )
    return scores


def cross_validate(
    dataset: ProjectDataset,
    config: ProjectConfig,
    *,
    task_name: str,
    fit_predict: FoldPredictor,
) -> pd.DataFrame:
    """Score ``fit_predict(train_rows, validation_rows)`` on development folds.

    ``fit_predict`` receives full observation frames so that baselines may use
    columns, such as the neighborhood group, that are not model features. It
    returns predictions (regression) or positive-class probabilities
    (classification) for the validation rows.
    """

    task = dataset.tasks[task_name]
    if task.target_column is None:
        raise ValueError("cross_validate requires a supervised task")
    development = time_partition(dataset.observations, config).development
    development = development.reset_index(drop=True)
    rows = []
    for fold, (train_index, valid_index) in enumerate(_folds(config, len(development))):
        train = development.iloc[train_index]
        valid = development.iloc[valid_index]
        output = np.asarray(fit_predict(train, valid))
        observed = valid[task.target_column].to_numpy()
        if task_name == "regression":
            scores = regression_scores(observed, output)
        else:
            probability = output.astype(float)
            scores = {
                "balanced_accuracy": float(
                    balanced_accuracy_score(observed, probability >= 0.5)
                ),
                "brier": float(brier_score_loss(observed, probability)),
            }
        rows.append({"fold": fold, **scores})
    return pd.DataFrame(rows)


def summarize(folds: pd.DataFrame) -> dict[str, float]:
    """Return mean and standard deviation for each fold metric."""

    summary: dict[str, float] = {}
    for column in folds.columns.drop("fold"):
        summary[f"{column}_mean"] = float(folds[column].mean())
        summary[f"{column}_std"] = float(folds[column].std(ddof=1))
    return summary


def _overall_median(train: pd.DataFrame, valid: pd.DataFrame) -> NDArray[np.float64]:
    return np.full(len(valid), float(train["sale_price"].median()))


def _neighborhood_median(
    train: pd.DataFrame, valid: pd.DataFrame
) -> NDArray[np.float64]:
    medians = train.groupby("neighborhood_group")["sale_price"].median()
    fallback = float(train["sale_price"].median())
    return valid["neighborhood_group"].map(medians).fillna(fallback).to_numpy(float)


def _majority_probability(
    train: pd.DataFrame, valid: pd.DataFrame
) -> NDArray[np.float64]:
    # A constant predictor: majority label for accuracy, base rate for Brier.
    rate = float(train["neighborhood_premium"].mean())
    return np.full(len(valid), rate)


def baseline_report(dataset: ProjectDataset, config: ProjectConfig) -> pd.DataFrame:
    """Cross-validate the documented baselines on development rows only."""

    baselines = (
        ("regression", "overall_median", _overall_median),
        ("regression", "neighborhood_median", _neighborhood_median),
        ("classification", "base_rate", _majority_probability),
    )
    rows = []
    for task_name, name, predictor in baselines:
        folds = cross_validate(
            dataset, config, task_name=task_name, fit_predict=predictor
        )
        rows.append({"task": task_name, "baseline": name, **summarize(folds)})
    return pd.DataFrame(rows)
