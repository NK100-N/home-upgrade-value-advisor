# Home Upgrade Value Advisor

Helps a homeowner decide which renovation is worth it. For a described house
and a proposed upgrade, such as kitchen quality from Typical to Excellent, the
product estimates the home's sale value before and after the upgrade. It also
gives an uncertainty interval and shows the most comparable homes behind the
estimate.

Final project for Rice University CMOR 438 / INDE 577, built from the course
[project template](TEMPLATE_GUIDE.md).

> **Scope.** Estimates are associations learned from Ames, Iowa, home sales,
> 2006–2010. They are not causal guarantees, not appraisals, and not valid for
> other markets or current prices without retraining. See the
> [responsible-use boundary](docs/PRODUCT_CONTRACT.md#responsible-use-boundary).

## Project documents

| Document | Contents |
| --- | --- |
| [Product contract](docs/PRODUCT_CONTRACT.md) | User, decision, prohibited uses, task-algorithm matrix, evaluation contract, system interfaces |
| [Dataset card](docs/DATASET_CARD.md) | Ames Housing provenance, checksums, filters, schema, leakage risks, limitations |
| [Scope decision](docs/decisions/0001-project-scope.md) | Why this project, and the alternatives rejected |
| [Team ownership](docs/TEAM_OWNERSHIP.md) | Ownership and maintenance responsibility |
| [Model card](docs/MODEL_CARD.md) | Selected model and its evidence (Milestone 5) |
| [Operations](docs/OPERATIONS.md) | Logs, monitoring, rollback, incident response (Milestone 7) |

## Machine-learning tasks

- **Regression:** predict sale price. The upgrade estimate is the difference
  between predictions with and without the upgrade.
- **Classification:** predict whether a home sells above its neighborhood's
  median price.
- **Unsupervised:** find stable home segments from price-free attributes, with
  PCA for representation.

Course algorithms are implemented from scratch in
[`src/dsm_project/algorithms/student/`](src/dsm_project/algorithms/student/)
and verified against scikit-learn as a labeled reference.

## Setup

Requires Git and [uv](https://docs.astral.sh/uv/). From the repository root:

```bash
uv sync --locked
uv run python scripts/setup_project.py
uv run dsm-project verify
```

Run the quality gate before every pull request:

```bash
uv run ruff check src tests scripts
uv run pytest -q
uv build
```

The Ames data are downloaded to the git-ignored `data/local/` folder and are
never committed. The download script and the project's data configuration
arrive in Milestone 2.

## Status

| Milestone | Status |
| --- | --- |
| 0. Repository and team contract | Template verified; branch protection on |
| 1. Product and data contract | In review (PR #8) |
| 2. Data boundary and baselines | Next |
| 2A–7. Algorithms, experiments, product, operations | Planned |

See [`MILESTONES.md`](MILESTONES.md) for the full sequence and
[`PROJECT_CHECKLIST.md`](PROJECT_CHECKLIST.md) for the definition of done.
