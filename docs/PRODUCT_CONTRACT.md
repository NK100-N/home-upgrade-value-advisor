# Product contract

Status: proposal (Milestone 1). Claims marked **open** need instructor or
reviewer confirmation before Milestone 2 begins.

## Users and decisions

**Intended user.** A homeowner planning a renovation on a limited budget.

**Decision supported.** Which candidate upgrade (for example, kitchen quality
from Typical to Excellent, a finished basement, an added bathroom) is associated
with the largest change in estimated market value *for this particular house*.
The homeowner compares that estimate with contractor quotes, which the product
does not produce.

**What the product returns.** For a described house and a proposed upgrade:

- estimated sale value before and after the upgrade;
- the estimated difference, with an uncertainty interval;
- the most comparable homes in the data (nearest neighbors) with and without the
  upgrade, so the user can inspect the evidence behind the estimate.

**Decisions the product must not automate.**

- Pricing a real sale, listing, appraisal, tax assessment, loan, or insurance
  decision.
- Telling a user that a renovation will *cause* a value increase. Estimates are
  associations learned from observed sales (see Responsible-use boundary).
- Estimating renovation cost, structural safety, or code compliance.

## Dataset suite

One dataset; all three tasks come from it. Details are in
[`DATASET_CARD.md`](DATASET_CARD.md).

| Property | Value |
| --- | --- |
| Source | Ames Housing (De Cock, 2011, *Journal of Statistics Education* 19(3)), built from Ames, Iowa Assessor's Office records |
| License | No explicit license in the published documentation; distributed by JSE for teaching. **Open:** confirm acceptable use with instructor |
| Observational unit | One residential property sale |
| Sampling process | All recorded residential sales in Ames, IA, 2006–July 2010 (2,930 sales) |
| Modeling population | Normal sales of residential-zoned properties with at most 4,000 sq ft above-ground living area (2,396 sales); exclusions justified in the dataset card |
| Time boundary | Sales 2006–2009 for development; 2010 sales (January–July) sealed as final test |
| Feature availability | Physical, quality, and location attributes of the house, all known before a sale. Sale-process fields are excluded |
| Targets | `SalePrice` (regression); premium-within-neighborhood event (classification); no target (unsupervised) |
| Representativeness limits | One small Midwestern city, 2006–2010, including the 2008 housing downturn. Not evidence about other markets or current prices |

## Task definitions

**Regression.** Predict `SalePrice` in US dollars from house attributes. Models
may fit on log price; reported errors are in dollars. The upgrade estimate is the
difference between predictions for the original attributes and the same
attributes with the upgraded feature changed.

**Classification.** Predict whether a home sells above the median price of its
neighborhood (positive event: "neighborhood premium"). The neighborhood medians
are computed from training rows only. This asks which non-location attributes
(quality, size, age, condition) move a home into the upper half of its own
market, which is the question a renovation decision depends on. **Open:**
reviewer to confirm this target is preferred over a fixed price threshold.

**Unsupervised.** Do Ames homes form stable segments (for example, older modest
homes, newer large homes, renovated older homes), and is the estimated value of
an upgrade different across segments? Price is excluded from clustering inputs
and used only afterward to describe segments.

## Task-algorithm matrix

All algorithms are student implementations under
`src/dsm_project/algorithms/student/`, verified on analytic cases and compared
with a labeled scikit-learn reference.

| Course family | Project task | Applicable? | Mathematical justification | Evidence |
| --- | --- | --- | --- | --- |
| Linear regression | Price regression | Yes | Price is approximately linear in size and ordinal quality on the log scale. Ridge penalty handles correlated size features. Coefficients give an interpretable per-upgrade association | Notebook, cross-validated RMSE vs. baselines |
| Logistic regression | Neighborhood-premium classification | Yes | Binary event. Log-odds linear in standardized attributes gives calibrated probabilities and interpretable coefficients | Notebook, balanced accuracy, calibration |
| Nearest neighbors | Regression, classification, and comparable-home retrieval | Yes | Mirrors appraisal "comps": nearby homes in standardized feature space have similar prices. Requires feature scaling and a chosen distance | Notebook, error vs. k, retrieved comparables in the product |
| Support vector machines and kernels | Premium classification | Yes | About 1,800 training rows keeps kernel methods tractable. An RBF kernel tests whether a nonlinear boundary beats logistic regression | Notebook, comparison with logistic regression |
| Decision trees | Regression and classification | Yes | Captures thresholds and interactions (for example, quality mattering more in some neighborhoods) without manual feature crosses | Notebook, pruning evidence |
| Ensembles | Regression and classification | Yes | Random forest and boosting reduce the variance of single trees. Expected strongest candidates on tabular data | Notebook, candidate comparison |
| Neural networks | Price regression | Yes, as a comparison | A small multilayer network tests whether learned nonlinear combinations improve on ensembles. Not expected to win on about 2,400 rows; the result is still evidence | Notebook, learning curves |
| Clustering | Home segmentation | Yes | k-means on standardized, price-free attributes defines segments. Evaluated by stability, not by agreement with price | Notebook, bootstrap stability, silhouette |
| Dimensionality reduction | Representation and visualization | Yes | PCA summarizes about 30 correlated size and quality features. Retained variance and reconstruction error show how many directions matter | Notebook, explained variance |

No family is excluded. Any family that fails verification or is not finished in
time will be recorded here as an exclusion with its reason.

## Evaluation contract

Fixed before any final-test result is inspected.

- **Partitions.** Development set: 2006–2009 sales. Model choice uses 5-fold
  cross-validation within it. Final test: 2010 sales, opened once after
  selection. The time split mirrors real use: predicting sales the model has not
  seen yet.
- **Regression metrics.** RMSE and MAE in dollars (primary: RMSE), plus RMSE on
  log price.
- **Classification metrics.** Balanced accuracy (primary), ROC AUC, and Brier
  score for calibration.
- **Unsupervised evidence.** Bootstrap stability (adjusted Rand index across
  resamples), silhouette score, and PCA explained variance and reconstruction
  error.
- **Baselines.** Regression: overall median price, and neighborhood median
  price. Classification: majority class. Every reported metric is compared with
  its baseline.
- **Uncertainty.** Fold-to-fold spread in cross-validation, and bootstrap
  intervals on final-test metrics. Prediction intervals for the product are
  calibrated on held-out development data.
- **Selection rule.** Lowest mean cross-validated RMSE. A simpler model is
  preferred if it is within one fold standard deviation of the best.
- **Slices.** Neighborhood, house age band, price tier, and kitchen-quality
  level.
- **Failure threshold.** A candidate that does not beat the neighborhood-median
  baseline is not promoted to the product.

## System interfaces

- **Data boundary.** A download script retrieves the published file into the
  git-ignored `data/local/` folder and verifies its SHA-256 checksum. An adapter
  in `src/dsm_project/data/` applies the documented filters and schema checks.
  CI uses a small constructed fixture in `data/fixtures/`.
- **Training command.** A `dsm-project` CLI command trains the candidates from
  a TOML configuration and writes the results.
- **Experiment artifact.** All candidate metrics are retained, plus a versioned
  model bundle with its configuration and data checksum.
- **Prediction API.** A FastAPI endpoint takes house attributes and a proposed
  upgrade, and returns the before and after estimates, the difference, the
  interval, and the comparable homes.
- **User interface.** A simple web form served with the API.
- **Deployment.** The template's Docker image.
- **Operations.** Structured request logs, latency and error metrics, an
  input-drift check against training distributions, and rollback to the
  previous model bundle (detailed in `OPERATIONS.md` at Milestone 7).

## Responsible-use boundary

- **Association, not causation.** Homes with excellent kitchens differ from
  other homes in many ways. A model can hold other recorded features fixed, but
  unrecorded factors remain. The interface states this beside every estimate.
- **Geography and time.** The estimates describe Ames, Iowa, from 2006 to 2010.
  The product must not be presented as valid elsewhere or today without
  retraining on local, current data.
- **Uncertainty.** Every estimate is shown with an interval. Inputs outside the
  training range (for example, very large or very old homes) are flagged.
- **Privacy.** The data are public assessor records with parcel IDs. Parcel IDs
  are not used as features, not shown in the interface, and not committed. The
  product stores no user-entered home details beyond request logs, which
  exclude addresses.
- **Fairness.** The data have no demographic fields. Neighborhood can act as a
  proxy for socioeconomic and historical patterns. Error rates are reported by
  neighborhood, and the product informs a homeowner's own choices but does not
  rank or score people or areas.
- **What a successful demonstration does not prove.** Good test metrics show
  the method predicts Ames sale prices from 2010. They do not show that a given
  renovation will raise a given home's price.
