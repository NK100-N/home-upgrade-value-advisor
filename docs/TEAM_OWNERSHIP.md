# Team ownership and integration

## Participants

- @NK100-N (solo project; **open:** confirm the solo arrangement and how
  pull-request review is assigned with the instructor)

## Shared core

As a solo project, @NK100-N owns the shared core:

- product contract and dataset card;
- data boundary;
- experiment pipeline;
- model selection and promotion;
- API, interface, and deployment;
- operations.

Pull requests go through branch protection and CI. Independent review is
requested as the course specifies.

## Individual vertical slices

**@NK100-N: renovation value estimation and comparable-home evidence.**

- **User need:** a homeowner comparing renovation options for their own house.
- **Owned interfaces:**
  - Ames data adapter and filters;
  - student implementations under `src/dsm_project/algorithms/student/`;
  - upgrade-difference estimator;
  - comparable-home retrieval;
  - prediction API and web form.
- **Tests, documentation, and operations:** recorded per pull request.

## Integration and conflict history

To be recorded as decisions arise. The first candidate is the choice of
classification target (neighborhood premium vs. a fixed price threshold); see
[`decisions/0001-project-scope.md`](decisions/0001-project-scope.md).

## Maintenance responsibility

@NK100-N investigates defects in every component through final review.
