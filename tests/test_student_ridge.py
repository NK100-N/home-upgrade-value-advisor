"""Mathematical, differential, and boundary tests for student ridge regression."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from dsm_project.algorithms import create_student_algorithm
from dsm_project.algorithms.student.linear import (
    RidgeRegression,
    ridge_gradient,
    ridge_objective,
)
from dsm_project.config import load_config
from dsm_project.data import load_project_dataset
from dsm_project.evaluation import cross_validate, model_fit_predict, summarize
from dsm_project.experiments import evaluate_supervised_estimator

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(60, 4)) * [1.0, 10.0, 0.1, 3.0]
    y = x @ [2.0, -0.5, 30.0, 1.0] + 7.0 + rng.normal(scale=0.3, size=60)
    return x, y


def test_recovers_an_exact_affine_rule_without_penalty() -> None:
    x = np.array([[0.0], [1.0], [2.0], [3.0]])
    model = RidgeRegression(alpha=0.0).fit(x, 1.5 + 2.0 * x[:, 0])
    np.testing.assert_allclose(model.coef_, [2.0], atol=1e-12)
    assert model.intercept_ == pytest.approx(1.5, abs=1e-12)


def test_hand_computed_penalized_slope() -> None:
    # Centered x = (-1, 0, 1), centered y = (-2, 0, 2): w = (x'y)/(x'x + a) = 4/3.
    x = np.array([[0.0], [1.0], [2.0]])
    y = np.array([1.0, 3.0, 5.0])
    model = RidgeRegression(alpha=1.0, standardize=False).fit(x, y)
    assert model.coef_[0] == pytest.approx(4.0 / 3.0)
    assert model.intercept_ == pytest.approx(3.0 - 4.0 / 3.0)


def test_gradient_vanishes_at_the_solution(data) -> None:
    x, y = data
    model = RidgeRegression(alpha=2.5, standardize=False).fit(x, y)
    grad_w, grad_b = ridge_gradient(x, y, model.coef_, model.intercept_, 2.5)
    np.testing.assert_allclose(grad_w, 0.0, atol=1e-8)
    assert grad_b == pytest.approx(0.0, abs=1e-8)


def test_gradient_matches_finite_differences(data) -> None:
    x, y = data
    rng = np.random.default_rng(1)
    w, b, alpha, step = rng.normal(size=4), 0.7, 1.3, 1e-6
    grad_w, grad_b = ridge_gradient(x, y, w, b, alpha)
    for j in range(4):
        e = np.zeros(4)
        e[j] = step
        numeric = (
            ridge_objective(x, y, w + e, b, alpha)
            - ridge_objective(x, y, w - e, b, alpha)
        ) / (2 * step)
        assert grad_w[j] == pytest.approx(numeric, rel=1e-5)
    numeric_b = (
        ridge_objective(x, y, w, b + step, alpha)
        - ridge_objective(x, y, w, b - step, alpha)
    ) / (2 * step)
    assert grad_b == pytest.approx(numeric_b, rel=1e-5)


@pytest.mark.parametrize("alpha", [0.0, 0.1, 10.0])
def test_agrees_with_trusted_ridge_on_raw_features(data, alpha: float) -> None:
    # Same objective and conventions: sum of squares, unpenalized intercept.
    x, y = data
    ours = RidgeRegression(alpha=alpha, standardize=False).fit(x, y)
    reference = Ridge(alpha=alpha, solver="svd").fit(x, y)
    np.testing.assert_allclose(ours.coef_, reference.coef_, rtol=1e-8, atol=1e-10)
    assert ours.intercept_ == pytest.approx(reference.intercept_, rel=1e-8)


def test_standardized_fit_predicts_like_reference_on_scaled_inputs(data) -> None:
    x, y = data
    scale = x.std(axis=0)
    ours = RidgeRegression(alpha=3.0).fit(x, y)
    reference = Ridge(alpha=3.0, solver="svd").fit(x / scale, y)
    np.testing.assert_allclose(ours.predict(x), reference.predict(x / scale), rtol=1e-8)


def test_penalty_shrinks_weights_monotonically(data) -> None:
    x, y = data
    norms = [
        np.linalg.norm(RidgeRegression(alpha=a).fit(x, y).coef_ * x.std(axis=0))
        for a in (0.0, 1.0, 10.0, 100.0, 1000.0)
    ]
    assert all(
        later < earlier for earlier, later in zip(norms, norms[1:], strict=False)
    )


def test_collinear_features_have_a_unique_penalized_solution() -> None:
    rng = np.random.default_rng(2)
    a = rng.normal(size=40)
    x = np.column_stack([a, 2 * a])
    model = RidgeRegression(alpha=1.0).fit(x, 3 * a)
    assert np.isfinite(model.coef_).all()
    unpenalized = RidgeRegression(alpha=0.0).fit(x, 3 * a)
    np.testing.assert_allclose(unpenalized.predict(x), 3 * a, atol=1e-8)


def test_constant_column_does_not_produce_nan() -> None:
    x = np.column_stack([np.arange(5.0), np.ones(5)])
    model = RidgeRegression(alpha=1.0).fit(x, np.arange(5.0))
    assert np.isfinite(model.predict(x)).all()


@pytest.mark.parametrize("alpha", [-1.0, float("nan"), float("inf")])
def test_rejects_invalid_penalty(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha"):
        RidgeRegression(alpha=alpha)


def test_rejects_bad_inputs_and_use_before_fit(data) -> None:
    x, y = data
    with pytest.raises(RuntimeError, match="fit must be called"):
        RidgeRegression().predict(x)
    with pytest.raises(ValueError, match="two-dimensional"):
        RidgeRegression().fit(x[:, 0], y)
    with pytest.raises(ValueError, match="one value per observation"):
        RidgeRegression().fit(x, y[:-1])
    bad = x.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        RidgeRegression().fit(bad, y)
    model = RidgeRegression().fit(x, y)
    with pytest.raises(ValueError, match="expected 4 features"):
        model.predict(x[:, :3])


def test_fit_is_deterministic(data) -> None:
    x, y = data
    first = RidgeRegression(alpha=0.5).fit(x, y)
    second = RidgeRegression(alpha=0.5).fit(x, y)
    np.testing.assert_array_equal(first.coef_, second.coef_)


def test_registry_and_configured_experiment_path() -> None:
    model = create_student_algorithm("ridge_regression", alpha=2.0)
    assert isinstance(model, RidgeRegression)

    config = load_config(ROOT / "configs" / "ames_fixture.toml")
    dataset = load_project_dataset(config)
    result = evaluate_supervised_estimator(
        dataset,
        config,
        task_name="regression",
        estimator_name="student:ridge_regression",
        estimator=model,
    )
    assert np.isfinite(result.metric_value)

    task = dataset.tasks["regression"]
    folds = cross_validate(
        dataset,
        config,
        task_name="regression",
        fit_predict=model_fit_predict(
            lambda: RidgeRegression(alpha=2.0),
            task.feature_columns,
            task.target_column,
            log_target=True,
        ),
    )
    assert summarize(folds)["rmse_mean"] > 0
