"""Write the constructed CI fixture that mirrors the Ames raw schema.

Every value is invented. No row is sampled from the real Ames records, because
their redistribution terms are unstated. The fixture deliberately includes rows
that the population filters must remove and "NA" values that mean "absent".

Usage: ``uv run python scripts/build_ames_fixture.py``
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "fixtures"
    / "ames_constructed.tsv"
)
NEIGHBORHOODS = ("NAmes", "CollgCr", "OldTown", "Edwards", "Somerst", "Tiny")
QUALITY = ("Fa", "TA", "Gd", "Ex")


def build(rows_per_year: int = 30, seed: int = 438) -> pd.DataFrame:
    """Return invented sales with the raw Ames column names."""

    rng = np.random.default_rng(seed)
    records = []
    order = 1
    for year in range(2006, 2011):
        for _ in range(rows_per_year):
            # "Tiny" is rare so that neighborhood pooling is exercised.
            neighborhood = NEIGHBORHOODS[
                rng.choice(len(NEIGHBORHOODS), p=[0.25, 0.25, 0.2, 0.15, 0.13, 0.02])
            ]
            quality = int(rng.integers(3, 10))
            area = float(rng.integers(700, 3200))
            basement = float(rng.choice([0.0, rng.integers(400, 1800)]))
            built = int(rng.integers(1900, 2006))
            kitchen = QUALITY[
                min(3, max(0, quality // 3 - 1 + int(rng.integers(0, 2))))
            ]
            fireplaces = int(rng.integers(0, 3))
            price = (
                20_000
                + 55 * area
                + 9_000 * quality
                + 20 * basement
                + 300 * (built - 1900)
                + float(rng.normal(0, 12_000))
            )
            records.append(
                {
                    "Order": order,
                    "PID": f"{900000000 + order:010d}",
                    "MS Zoning": "RL",
                    "Neighborhood": neighborhood,
                    "Central Air": "Y" if rng.random() > 0.1 else "N",
                    "Yr Sold": year,
                    "Sale Condition": "Normal",
                    "SalePrice": round(max(price, 35_000)),
                    "Overall Qual": quality,
                    "Overall Cond": int(rng.integers(3, 9)),
                    "Year Built": built,
                    "Year Remod/Add": int(rng.integers(built, 2007)),
                    "Gr Liv Area": area,
                    "Total Bsmt SF": basement,
                    "1st Flr SF": area * 0.6,
                    "2nd Flr SF": area * 0.4,
                    "Lot Area": float(rng.integers(4_000, 20_000)),
                    "Mas Vnr Area": "NA" if rng.random() < 0.1 else 0.0,
                    "Garage Cars": int(rng.integers(0, 4)),
                    "Garage Area": float(rng.integers(0, 900)),
                    "Full Bath": int(rng.integers(1, 4)),
                    "Half Bath": int(rng.integers(0, 2)),
                    "Bsmt Full Bath": "NA"
                    if basement == 0
                    else int(rng.integers(0, 2)),
                    "Bsmt Half Bath": "NA" if basement == 0 else 0,
                    "Bedroom AbvGr": int(rng.integers(1, 6)),
                    "TotRms AbvGrd": int(rng.integers(4, 11)),
                    "Fireplaces": fireplaces,
                    "Wood Deck SF": float(rng.integers(0, 400)),
                    "Open Porch SF": float(rng.integers(0, 200)),
                    "Kitchen Qual": kitchen,
                    "Exter Qual": QUALITY[int(rng.integers(1, 4))],
                    "Bsmt Qual": "NA"
                    if basement == 0
                    else QUALITY[int(rng.integers(1, 4))],
                    "Heating QC": QUALITY[int(rng.integers(0, 4))],
                    "Fireplace Qu": "NA" if fireplaces == 0 else "TA",
                }
            )
            order += 1
    frame = pd.DataFrame(records)
    # Rows the population filters must remove, one per rule.
    frame.loc[0, "Sale Condition"] = "Abnorml"
    frame.loc[1, "MS Zoning"] = "C (all)"
    frame.loc[2, "Gr Liv Area"] = 4_500.0
    return frame


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    build().to_csv(OUTPUT, sep="\t", index=False)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
