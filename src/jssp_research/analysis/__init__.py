"""Benchmark suite selection and statistical analysis helpers."""

from __future__ import annotations

from jssp_research.analysis.statistics import (
    FriedmanResult,
    bootstrap_median_diff_ci,
    cliffs_delta,
    holm_bonferroni_adjust,
    lower_bound_gap,
    paired_comparisons_table,
    relative_percentage_deviation,
    run_friedman_analysis,
    summarize_numeric,
    vargha_delaney_a12,
)

__all__ = [
    "FriedmanResult",
    "bootstrap_median_diff_ci",
    "cliffs_delta",
    "holm_bonferroni_adjust",
    "lower_bound_gap",
    "paired_comparisons_table",
    "relative_percentage_deviation",
    "run_friedman_analysis",
    "summarize_numeric",
    "vargha_delaney_a12",
]
