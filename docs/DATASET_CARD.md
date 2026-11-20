# Dataset card

## Identity and provenance

- **Dataset name:** Ames Housing (`AmesHousing.txt`).
- **Source owner:** Ames, Iowa Assessor's Office records, compiled by Dean
  De Cock. Published with "Ames, Iowa: Alternative to the Boston Housing Data
  Set," *Journal of Statistics Education* 19(3), 2011.
- **Retrieval method and date:** HTTPS download from
  `https://jse.amstat.org/v19n3/decock/AmesHousing.txt`, with documentation at
  `https://jse.amstat.org/v19n3/decock/DataDocumentation.txt`. Retrieved
  2026-11-16.
- **License or access agreement:** the published documentation states no
  explicit license. The file is distributed by the *Journal of Statistics
  Education* for teaching use. Redistribution terms are unstated, so the raw
  file is not committed: the download script retrieves it and CI uses
  a constructed fixture.
- **Immutable version and checksums (SHA-256):**
  - `AmesHousing.txt`:
    `6cfe6cb525ba437de428653a1040e2aed7d696640bf75203786a6d7a0e67cfcc`
  - `DataDocumentation.txt`:
    `93d741d17422cca39584c6f03fd3f45194501101790bd8241e33d21c991e3eed`

## Observation and sampling

- **What one row represents:** one sale of one residential property, described
  by the assessor's record for that property.
- **What produced the rows:** all residential sales recorded by the Ames
  Assessor's Office from January 2006 through July 2010. The source file has
  2,930 rows and 82 columns (23 nominal, 23 ordinal, 14 discrete, and
  20 continuous variables, plus 2 identifiers).
- **Repetition and dependence:** parcel IDs do not repeat, so each property
  appears once. Sales in the same neighborhood and year share local market
  conditions, so errors may be correlated within neighborhoods.
- **Boundaries that matter:**
  - **Time.** Sales span 2006–2010, including the 2008 housing downturn. Rows by
    year: 2006: 625, 2007: 694, 2008: 622, 2009: 648, 2010: 341 (January–July
    only).
  - **Space.** 28 neighborhoods inside one city.

## Modeling population

These filters are applied in the data adapter and are reported, not hidden.

| Filter | Rows removed | Reason |
| --- | --- | --- |
| `Sale Condition` is not `Normal` | 517 | Partial (new construction sold before completion), abnormal (foreclosure or short sale), family, allocation, and adjoining-land sales do not reflect typical market value. The dataset author recommends this for a "typical sale" model |
| `MS Zoning` is commercial, industrial, or agricultural | 16 | Not residential homes, which is the product's scope |
| `Gr Liv Area` above 4,000 sq ft | 1 (the other 4 of the author's 5 unusual sales were already removed as non-normal) | The author identifies these as partial sales or very unusual homes |

Filters apply in the order listed; counts are rows removed at each step.

Rows remaining: 2,396. Development (2006–2009): 2,095. Final test (2010): 301.

Several neighborhoods have very few sales after filtering (Landmrk 1,
GrnHill 2, Greens 8, Blueste 10). Neighborhoods with fewer than 20 training
sales are pooled into an "other" level for the neighborhood-median target and
features, and their errors are reported separately.

## Schema and measurement

- **Identifiers:**
  - `Order`: row number.
  - `PID`: parcel ID. Never used as a feature.
- **Target:** `SalePrice`, the sale price in US dollars. Median $160,000; range
  $12,789–$755,000 before filtering. Right-skewed, and its spread grows with
  price (the dataset author notes heteroscedasticity).
- **Feature groups, all known before a sale:**
  - **Size:** `Gr Liv Area`, `Total Bsmt SF`, `1st Flr SF`, `2nd Flr SF`,
    `Lot Area`, `Garage Area` (square feet); `Garage Cars`.
  - **Rooms:** `Full Bath`, `Half Bath`, `Bsmt Full Bath`, `Bedroom AbvGr`,
    `TotRms AbvGrd`, `Fireplaces`.
  - **Quality and condition:**
    - `Overall Qual` and `Overall Cond` are integers from 1 to 10.
    - `Kitchen Qual`, `Exter Qual`, `Bsmt Qual`, `Heating QC`, and
      `Fireplace Qu` use the ordinal scale Po < Fa < TA < Gd < Ex.
    - **Kitchen Qual** is the main upgrade lever. Its distribution: TA 1,494,
      Gd 1,160, Ex 205, Fa 70, Po 1.
  - **Age:** `Year Built`, `Year Remod/Add` (equal to `Year Built` when never
    remodeled; 46.5% of homes show a remodel).
  - **Location and type:** `Neighborhood` (28 levels), `MS Zoning`,
    `Bldg Type`, `House Style`.
- **Missing-value convention:** in quality fields, `NA` usually means the
  feature does not exist rather than being unknown. For example:
  - `Fireplace Qu` is missing for the 1,422 homes without a fireplace.
  - `Pool QC` is missing for 2,917 homes without a pool.
  - Basement fields are missing for about 80 homes without a basement.

  These are encoded as an explicit "none" level. Missing `Lot Frontage`
  (490 rows) is truly unknown and is imputed inside the fitted pipeline.
- **Leakage risks, excluded from features:**
  - `Sale Type`, `Sale Condition`, `Mo Sold`, and `Yr Sold` describe the sale,
    not the house. `Yr Sold` defines the split instead.
  - `Misc Val` and `Pool QC` are too sparse to support claims.
  - Neighborhood medians used by the classification target are computed on
    training rows only.

## Data access and privacy

- **Committed:** this card, the download script and checksums, and a small
  constructed fixture in `data/fixtures/`. The fixture uses invented values that
  follow the real schema and missing-value patterns. It is not sampled from
  real records, because the redistribution terms are unconfirmed.
- **Local only:** the raw file in `data/local/`, which Git ignores.
- **Credentials:** none; the source is a public HTTPS URL.
- **Privacy:** the records are public tax-assessment data and contain no owner
  names. Parcel IDs could link a row to an address on the assessor's website,
  so they are dropped at ingestion and never displayed.

## Representativeness and limitations

- **Geography and time:** one small university city in Iowa, 2006–2010. Prices,
  housing stock, and renovation styles differ from other regions, such as
  Houston, and from current markets.
- **Market shock:** the 2008 downturn falls inside the development period, and
  2010 is a partial year. The time-based test measures how well the model holds
  up as the market shifts.
- **Quality ratings are assessor judgments,** not standardized inspections, and
  may be inconsistent between assessors or years.
- **No cost or causal information:** the data record what homes sold for, not
  what renovations cost or what would have happened without them.
- **Not evidence for excluded sales:** foreclosures, family transfers, new
  construction, very large homes, and non-residential properties are outside
  the modeling population.
