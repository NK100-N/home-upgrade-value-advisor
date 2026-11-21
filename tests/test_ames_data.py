"""Contracts for the Ames data boundary, partitions, and baselines.

These tests run on the constructed fixture, never on the real Ames file.
"""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dsm_project.config import ProjectConfig, load_config
from dsm_project.data import load_project_dataset
from dsm_project.data.ames import (
    MIN_NEIGHBORHOOD_SALES,
    POOLED_NEIGHBORHOOD,
    STRUCTURE_FEATURES,
    read_raw,
)
from dsm_project.data.splits import kfold_indices, time_partition
from dsm_project.evaluation import baseline_report
from dsm_project.experiments import fit_supervised_candidate

ROOT = Path(__file__).resolve().parent.parent
FIXTURE_CONFIG = ROOT / "configs" / "ames_fixture.toml"
FIXTURE = ROOT / "data" / "fixtures" / "ames_constructed.tsv"


def _copy_project(tmp_path: Path, raw: pd.DataFrame) -> ProjectConfig:
    """Write a modified fixture and its configuration into a temporary root."""

    (tmp_path / "configs").mkdir()
    (tmp_path / "data" / "fixtures").mkdir(parents=True)
    shutil.copy(FIXTURE_CONFIG, tmp_path / "configs" / "ames_fixture.toml")
    raw.to_csv(
        tmp_path / "data" / "fixtures" / "ames_constructed.tsv", sep="\t", index=False
    )
    return load_config(tmp_path / "configs" / "ames_fixture.toml")


@pytest.fixture(scope="module")
def config() -> ProjectConfig:
    return load_config(FIXTURE_CONFIG)


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    return read_raw(FIXTURE)


def test_fixture_matches_its_generator() -> None:
    spec = importlib.util.spec_from_file_location(
        "build_ames_fixture", ROOT / "scripts" / "build_ames_fixture.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = read_raw(FIXTURE)
    rebuilt = module.build()
    assert len(rebuilt) == len(expected)
    assert list(rebuilt.columns) == list(expected.columns)
    assert rebuilt["SalePrice"].tolist() == expected["SalePrice"].tolist()


def test_population_filters_remove_one_row_per_rule(
    config: ProjectConfig, raw: pd.DataFrame
) -> None:
    dataset = load_project_dataset(config)
    assert len(dataset.observations) == len(raw) - 3
    removed_orders = set(raw["Order"]) - set(dataset.observations["order"])
    assert removed_orders == {1, 2, 3}


def test_parcel_identifier_never_reaches_the_dataset(config: ProjectConfig) -> None:
    observations = load_project_dataset(config).observations
    assert not any("pid" in column.lower() for column in observations.columns)


def test_absent_features_are_encoded_as_zero(
    config: ProjectConfig, raw: pd.DataFrame
) -> None:
    observations = load_project_dataset(config).observations.set_index("order")
    no_fireplace = raw.loc[raw["Fireplace Qu"].isna(), "Order"]
    no_fireplace = no_fireplace[no_fireplace.isin(observations.index)]
    assert (observations.loc[no_fireplace, "fireplace_qu"] == 0).all()
    assert np.isfinite(observations[list(STRUCTURE_FEATURES)].to_numpy()).all()
    assert set(observations["central_air"].unique()) <= {0.0, 1.0}


def test_rare_neighborhoods_are_pooled_on_development_counts(
    config: ProjectConfig,
) -> None:
    dataset = load_project_dataset(config)
    development = time_partition(dataset.observations, config).development
    counts = development["neighborhood"].value_counts()
    rare = set(counts[counts < MIN_NEIGHBORHOOD_SALES].index)
    assert rare, "the fixture must exercise pooling"
    pooled = dataset.observations["neighborhood"].isin(rare)
    assert (
        dataset.observations.loc[pooled, "neighborhood_group"] == POOLED_NEIGHBORHOOD
    ).all()
    indicators = [
        c for c in dataset.tasks["regression"].feature_columns if c.startswith("nbhd_")
    ]
    assert (dataset.observations[indicators].sum(axis=1) == 1).all()


def test_premium_target_ignores_final_test_prices(
    tmp_path: Path, config: ProjectConfig, raw: pd.DataFrame
) -> None:
    original = load_project_dataset(config).observations
    shifted = raw.copy()
    shifted.loc[shifted["Yr Sold"] == 2010, "SalePrice"] *= 10
    changed = load_project_dataset(_copy_project(tmp_path, shifted)).observations

    def development_labels(frame: pd.DataFrame) -> pd.Series:
        return frame.loc[frame["yr_sold"] != 2010].set_index("order")[
            "neighborhood_premium"
        ]

    pd.testing.assert_series_equal(
        development_labels(original), development_labels(changed)
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda f: f.drop(columns=["Kitchen Qual"]), "missing required columns"),
        (lambda f: f.assign(Order=1), "unique"),
        (
            lambda f: f.assign(SalePrice=f["SalePrice"].where(f.index != 5, 0)),
            "positive",
        ),
        (
            lambda f: f.assign(**{"Yr Sold": f["Yr Sold"].replace(2006, 2015)}),
            "Yr Sold",
        ),
        (lambda f: f.assign(**{"Kitchen Qual": "Superb"}), "unknown quality"),
        (lambda f: f.assign(**{"Central Air": "Maybe"}), "Central Air"),
    ],
)
def test_invalid_raw_data_is_rejected(
    tmp_path: Path, raw: pd.DataFrame, mutate, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        load_project_dataset(_copy_project(tmp_path, mutate(raw.copy())))


def test_missing_data_file_explains_the_download_step(tmp_path: Path) -> None:
    (tmp_path / "configs").mkdir()
    text = FIXTURE_CONFIG.read_text().replace(
        "data/fixtures/ames_constructed.tsv", "data/local/AmesHousing.txt"
    )
    (tmp_path / "configs" / "ames.toml").write_text(text)
    with pytest.raises(FileNotFoundError, match="download_ames.py"):
        load_project_dataset(load_config(tmp_path / "configs" / "ames.toml"))


def test_time_partition_seals_the_final_year(config: ProjectConfig) -> None:
    observations = load_project_dataset(config).observations
    partition = time_partition(observations, config)
    assert set(partition.development["yr_sold"]) == {2006, 2007, 2008, 2009}
    assert set(partition.final_test["yr_sold"]) == {2010}
    assert len(partition.development) + len(partition.final_test) == len(observations)


def test_kfold_indices_partition_rows_reproducibly() -> None:
    folds = kfold_indices(23, 5, seed=1)
    validation = np.concatenate([valid for _, valid in folds])
    assert sorted(validation.tolist()) == list(range(23))
    for train, valid in folds:
        assert not set(train) & set(valid)
    again = kfold_indices(23, 5, seed=1)
    assert all((a[1] == b[1]).all() for a, b in zip(folds, again, strict=True))
    with pytest.raises(ValueError):
        kfold_indices(3, 5, seed=1)


def test_reference_holdout_never_uses_final_test_rows(config: ProjectConfig) -> None:
    dataset = load_project_dataset(config)
    result = fit_supervised_candidate(
        dataset, config, task_name="regression", model_name="linear_regression"
    )
    development = time_partition(dataset.observations, config).development
    assert result.train_rows + result.test_rows == len(development)


def test_baselines_use_development_rows_only(
    tmp_path: Path, config: ProjectConfig, raw: pd.DataFrame
) -> None:
    report = baseline_report(load_project_dataset(config), config)
    assert set(report["baseline"]) == {
        "overall_median",
        "neighborhood_median",
        "base_rate",
    }
    classification = report.loc[report["baseline"] == "base_rate"].iloc[0]
    assert classification["balanced_accuracy_mean"] == pytest.approx(0.5)

    shifted = raw.copy()
    shifted.loc[shifted["Yr Sold"] == 2010, "SalePrice"] *= 10
    shifted_config = _copy_project(tmp_path, shifted)
    shifted_report = baseline_report(
        load_project_dataset(shifted_config), shifted_config
    )
    pd.testing.assert_frame_equal(report, shifted_report)
