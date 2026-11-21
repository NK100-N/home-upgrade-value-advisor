# Data boundary

`data/local/` is ignored by Git and is the default location for student-supplied
files. Do not weaken this rule to make a notebook convenient.

## Ames Housing (project data)

Download and verify the published file:

```bash
uv run python scripts/download_ames.py
```

The script writes `AmesHousing.txt` and `DataDocumentation.txt` into
`data/local/` and checks each file's SHA-256 checksum against the values in
`docs/DATASET_CARD.md`. A mismatch stops the script: the published file has
changed and the dataset card must be reviewed before use. The raw file is never
committed, because its redistribution terms are unstated.

Load it through the configured boundary:

```bash
uv run dsm-project --config configs/ames.toml baselines
```

`src/dsm_project/data/ames.py` validates the raw file, applies the documented
population filters, drops the parcel ID, encodes the assessor quality scale,
and defines the three tasks. Neighborhood pooling and the neighborhood-premium
target are computed from development sales (2006–2009) only.

## CI fixture

`data/fixtures/ames_constructed.tsv` is **constructed**: every value is
invented by `scripts/build_ames_fixture.py`, and no row is sampled from the real
records. It mirrors the raw column names and missing-value conventions, and it
includes one row for each population filter to remove. `configs/ames_fixture.toml`
runs the full pipeline on it. Regenerate it with:

```bash
uv run python scripts/build_ames_fixture.py
```

Fixture results test schemas and transformations only. They are not evidence
about Ames sale prices.

## Template example

The template's reference configuration, `configs/example.toml`, uses the Wine
Recognition dataset packaged with scikit-learn, credited to the UCI Machine
Learning Repository. It loads without network access and is kept so that the
template's own tests continue to verify the reference path.
