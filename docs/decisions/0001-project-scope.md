# ADR 0001: Project scope

- Status: proposed
- Date: 2026-11-16

## Context

Homeowners choosing between renovations rely on generic rules of thumb or on
advice from contractors who benefit from any choice. The question that matters
to them is house-specific: for *this* house, which upgrade is associated with
the largest change in market value?

The final project must:

- be one coherent product;
- use student-authored implementations of the applicable course algorithms;
- cover supervised and unsupervised tasks;
- rely on lawfully accessible data with a CI-safe fixture;
- end in a deployed, observable service.

The project is solo and must fit the remaining semester.

## Decision

Build **Home Upgrade Value Advisor** on the Ames Housing dataset (2,396 normal
residential sales after documented filters).

- **Regression:** sale price. The upgrade estimate is the difference between
  predictions with and without the upgraded attribute.
- **Classification:** whether a home sells above its neighborhood's median
  price.
- **Unsupervised:** stable home segments from price-free attributes, with PCA.
- **Split:** 2006–2009 sales for development with 5-fold cross-validation;
  2010 sales as a sealed final test.
- **Product:** a FastAPI service and web form. They return the before and after
  estimates, an uncertainty interval, and comparable homes.

Full contract: [`../PRODUCT_CONTRACT.md`](../PRODUCT_CONTRACT.md).

## Alternatives considered

- **Photo-based renovation estimator** (measure a room from a picture, price
  materials from retailer websites). Rejected for three reasons:
  - measuring from images requires pretrained vision models rather than course
    algorithms;
  - retailer scraping raises terms-of-service concerns;
  - material prices give no learnable target.
- **Building-permit cost data plus Ames sale prices.** Rejected because permit
  costs and sale prices come from different cities and cannot be linked to the
  same homes.
- **Air quality, flood risk, or building energy.** Credible, but time-series
  leakage and multi-source joins (air quality, flood risk) or a small simulated
  dataset (energy, 768 rows) add risk without serving the renovation question.
- **Fixed price threshold for classification** (for example, above $200,000).
  Rejected: it mostly measures location and size. Neighborhood premium
  isolates the attributes a renovation can change.

## Consequences

- **Benefits:**
  - one well-documented dataset;
  - every course family applies without forcing;
  - the comparable-home retrieval gives users inspectable evidence.
- **Costs and risks:**
  - The dataset's redistribution terms are unstated, so the raw file stays
    local and CI uses a constructed fixture.
  - The upgrade estimates are associations, not causal effects.
  - Results describe Ames from 2006 to 2010 only.
  - Ames is a widely used teaching dataset, so originality must come from the
    decision framing and the depth of evidence.
- **Revisit if:**
  - the 2010 test set proves too small for stable conclusions;
  - the neighborhood-premium target proves degenerate in some neighborhoods.
    Several have few sales, so the medians may be unstable.
