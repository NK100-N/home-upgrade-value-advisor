"""Evaluation helpers that preserve task-specific meaning."""

from dsm_project.evaluation.comparison import comparison_table
from dsm_project.evaluation.cross_validation import (
    baseline_report,
    cross_validate,
    regression_scores,
    summarize,
)

__all__ = [
    "baseline_report",
    "comparison_table",
    "cross_validate",
    "regression_scores",
    "summarize",
]
