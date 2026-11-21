"""Ames Housing data boundary: ingestion, filters, encoding, and task targets.

The adapter turns the published tab-separated Ames file into validated,
model-ready observations. Every rule here is deterministic and documented in
``docs/DATASET_CARD.md``. Rules that depend on data values (neighborhood
pooling and the neighborhood-premium target) are computed from development
rows only, so the sealed final-test rows never influence them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from dsm_project.config import ProjectConfig
from dsm_project.contracts import (
    DatasetProvenance,
    ProjectDataset,
    TaskDefinition,
    TaskKind,
)

IDENTIFIER = "order"
TARGET = "sale_price"
PREMIUM_TARGET = "neighborhood_premium"
TIME_COLUMN = "yr_sold"
NEIGHBORHOOD = "neighborhood"

RESIDENTIAL_ZONES = frozenset({"RL", "RM", "FV", "RH"})
MAX_LIVING_AREA = 4_000
MIN_NEIGHBORHOOD_SALES = 20
POOLED_NEIGHBORHOOD = "Other"

# Assessor quality scale; "NA" in these fields means the feature is absent.
QUALITY_SCALE = {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}

NUMERIC_COLUMNS = {
    "Overall Qual": "overall_qual",
    "Overall Cond": "overall_cond",
    "Year Built": "year_built",
    "Year Remod/Add": "year_remod_add",
    "Gr Liv Area": "gr_liv_area",
    "Total Bsmt SF": "total_bsmt_sf",
    "1st Flr SF": "first_flr_sf",
    "2nd Flr SF": "second_flr_sf",
    "Lot Area": "lot_area",
    "Mas Vnr Area": "mas_vnr_area",
    "Garage Cars": "garage_cars",
    "Garage Area": "garage_area",
    "Full Bath": "full_bath",
    "Half Bath": "half_bath",
    "Bsmt Full Bath": "bsmt_full_bath",
    "Bsmt Half Bath": "bsmt_half_bath",
    "Bedroom AbvGr": "bedrooms",
    "TotRms AbvGrd": "total_rooms",
    "Fireplaces": "fireplaces",
    "Wood Deck SF": "wood_deck_sf",
    "Open Porch SF": "open_porch_sf",
}
QUALITY_COLUMNS = {
    "Kitchen Qual": "kitchen_qual",
    "Exter Qual": "exter_qual",
    "Bsmt Qual": "bsmt_qual",
    "Heating QC": "heating_qc",
    "Fireplace Qu": "fireplace_qu",
}
# Missing values in these counts and areas mean "no basement" or "no veneer".
ABSENT_MEANS_ZERO = ("mas_vnr_area", "bsmt_full_bath", "bsmt_half_bath")

REQUIRED_RAW_COLUMNS = (
    "Order",
    "PID",
    "MS Zoning",
    "Neighborhood",
    "Central Air",
    "Yr Sold",
    "Sale Condition",
    "SalePrice",
    *NUMERIC_COLUMNS,
    *QUALITY_COLUMNS,
)

# Physical attributes a renovation or a buyer can observe; no price, no location.
STRUCTURE_FEATURES = (
    *NUMERIC_COLUMNS.values(),
    *QUALITY_COLUMNS.values(),
    "central_air",
)


def read_raw(path: str | Path) -> pd.DataFrame:
    """Read the published tab-separated Ames file without interpretation."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(
            f"Ames data not found: {source}. Run "
            "`uv run python scripts/download_ames.py` first."
        )
    return pd.read_csv(source, sep="\t", keep_default_na=False, na_values=["NA", ""])


def apply_population_filters(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Keep normal residential sales and report rows removed by each rule."""

    rules = {
        "non_normal_sale": lambda f: f["Sale Condition"] != "Normal",
        "non_residential_zoning": lambda f: ~f["MS Zoning"].isin(RESIDENTIAL_ZONES),
        "living_area_over_4000": lambda f: f["Gr Liv Area"] > MAX_LIVING_AREA,
    }
    removed: dict[str, int] = {}
    kept = raw
    for name, rule in rules.items():
        mask = rule(kept)
        removed[name] = int(mask.sum())
        kept = kept.loc[~mask]
    return kept.copy(), removed


def validate_raw(raw: pd.DataFrame) -> None:
    """Reject structural and value defects before any encoding."""

    missing = [column for column in REQUIRED_RAW_COLUMNS if column not in raw]
    if missing:
        raise ValueError(f"Ames file is missing required columns: {missing}")
    if raw["Order"].isna().any() or not raw["Order"].is_unique:
        raise ValueError("Ames 'Order' identifiers must be present and unique")
    price = pd.to_numeric(raw["SalePrice"], errors="coerce")
    if price.isna().any() or (price <= 0).any():
        raise ValueError("SalePrice must be a positive number for every sale")
    if (pd.to_numeric(raw["Gr Liv Area"], errors="coerce") <= 0).any():
        raise ValueError("Gr Liv Area must be positive")
    years = pd.to_numeric(raw["Yr Sold"], errors="coerce")
    if years.isna().any() or not years.between(2006, 2010).all():
        raise ValueError("Yr Sold must lie between 2006 and 2010")
    for column in QUALITY_COLUMNS:
        unknown = set(raw[column].dropna()) - set(QUALITY_SCALE)
        if unknown:
            raise ValueError(f"{column} has unknown quality codes: {sorted(unknown)}")
    if not set(raw["Central Air"].dropna()) <= {"Y", "N"}:
        raise ValueError("Central Air must be 'Y' or 'N'")


def encode(filtered: pd.DataFrame) -> pd.DataFrame:
    """Rename, encode ordinal scales, and drop the parcel identifier."""

    frame = pd.DataFrame(
        {
            IDENTIFIER: filtered["Order"].astype(int).to_numpy(),
            TIME_COLUMN: filtered["Yr Sold"].astype(int).to_numpy(),
            NEIGHBORHOOD: filtered["Neighborhood"].astype(str).to_numpy(),
            TARGET: filtered["SalePrice"].astype(float).to_numpy(),
        }
    )
    for raw_name, name in NUMERIC_COLUMNS.items():
        frame[name] = pd.to_numeric(filtered[raw_name]).to_numpy(dtype=float)
    for name in ABSENT_MEANS_ZERO:
        frame[name] = frame[name].fillna(0.0)
    for raw_name, name in QUALITY_COLUMNS.items():
        codes = filtered[raw_name].map(QUALITY_SCALE).fillna(0)
        frame[name] = codes.to_numpy(dtype=float)
    frame["central_air"] = (filtered["Central Air"] == "Y").to_numpy(dtype=float)
    remaining = frame[list(STRUCTURE_FEATURES)].isna().sum()
    if remaining.any():
        raise ValueError(f"Unexplained missing values: {remaining[remaining > 0]}")
    return frame


def development_mask(frame: pd.DataFrame, final_test_values: list[int]) -> pd.Series:
    """Return True for rows that may inform fitted or data-derived rules."""

    return ~frame[TIME_COLUMN].isin(final_test_values)


def add_neighborhood_features(
    frame: pd.DataFrame, development: pd.Series
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Pool rare neighborhoods and add one indicator column per kept level.

    Rarity is judged on development rows only.
    """

    counts = frame.loc[development, NEIGHBORHOOD].value_counts()
    kept_levels = sorted(counts[counts >= MIN_NEIGHBORHOOD_SALES].index)
    grouped = frame[NEIGHBORHOOD].where(
        frame[NEIGHBORHOOD].isin(kept_levels), POOLED_NEIGHBORHOOD
    )
    frame = frame.assign(neighborhood_group=grouped)
    levels = [*kept_levels, POOLED_NEIGHBORHOOD]
    indicator_names = tuple(f"nbhd_{level.lower()}" for level in levels)
    indicators = pd.DataFrame(
        {
            name: (grouped == level).to_numpy(dtype=float)
            for name, level in zip(indicator_names, levels, strict=True)
        },
        index=frame.index,
    )
    return pd.concat([frame, indicators], axis=1), indicator_names


def add_premium_target(frame: pd.DataFrame, development: pd.Series) -> pd.DataFrame:
    """Label a sale 1 when it exceeds its neighborhood's development median."""

    medians = frame.loc[development].groupby("neighborhood_group")[TARGET].median()
    reference = frame["neighborhood_group"].map(medians)
    if reference.isna().any():
        raise ValueError("A neighborhood group has no development sales")
    return frame.assign(**{PREMIUM_TARGET: (frame[TARGET] > reference).astype(int)})


class AmesHousingAdapter:
    """Load Ames Housing as one validated project dataset with three tasks."""

    def load(self, config: ProjectConfig) -> ProjectDataset:
        """Read, validate, filter, encode, and declare the project tasks."""

        path = config.root / str(config.dataset.get("path", ""))
        raw = read_raw(path)
        validate_raw(raw)
        filtered, _ = apply_population_filters(raw)
        if filtered.empty:
            raise ValueError("No sales remain after the population filters")
        frame = encode(filtered)

        final_test_values = [int(v) for v in config.split.get("final_test_values", [])]
        development = development_mask(frame, final_test_values)
        if not development.any():
            raise ValueError("No development rows remain before the final test")
        frame, neighborhood_features = add_neighborhood_features(frame, development)
        frame = add_premium_target(frame, development)

        supervised_features = (*STRUCTURE_FEATURES, *neighborhood_features)
        tasks = {
            "regression": TaskDefinition(
                name=str(config.tasks["regression"]["name"]),
                kind=TaskKind.REGRESSION,
                feature_columns=supervised_features,
                target_column=TARGET,
                metric=str(config.tasks["regression"]["metric"]),
            ),
            "classification": TaskDefinition(
                name=str(config.tasks["classification"]["name"]),
                kind=TaskKind.CLASSIFICATION,
                feature_columns=supervised_features,
                target_column=PREMIUM_TARGET,
                metric=str(config.tasks["classification"]["metric"]),
            ),
            "unsupervised": TaskDefinition(
                name=str(config.tasks["unsupervised"]["name"]),
                kind=TaskKind.UNSUPERVISED,
                feature_columns=STRUCTURE_FEATURES,
                target_column=None,
                metric=str(config.tasks["unsupervised"]["metric"]),
            ),
        }
        dataset = ProjectDataset(
            observations=frame.reset_index(drop=True),
            identifier=IDENTIFIER,
            tasks=tasks,
            provenance=DatasetProvenance(
                name=str(config.dataset["name"]),
                source=str(config.dataset["source"]),
                license=str(config.dataset["license"]),
                observation_unit=str(config.dataset["observation_unit"]),
            ),
        )
        dataset.validate()
        if not np.isfinite(frame[list(supervised_features)].to_numpy()).all():
            raise ValueError("Model features must be finite after encoding")
        return dataset
