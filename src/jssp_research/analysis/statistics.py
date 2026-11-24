"""Shared statistical inference, effect sizes, RPD, and bootstrap confidence intervals."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any

import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon


def relative_percentage_deviation(makespan: float | int, bks: float | int) -> float:
    """Compute exact RPD (%) against reference BKS upper bound."""
    if bks <= 0:
        raise ValueError(f"BKS must be positive, got {bks}")
    return 100.0 * (float(makespan) - float(bks)) / float(bks)


def lower_bound_gap(makespan: float | int, lb: float | int | None) -> float | None:
    """Compute optimality gap (%) relative to lower bound when available."""
    if lb is None or lb <= 0:
        return None
    return 100.0 * (float(makespan) - float(lb)) / float(lb)


def summarize_numeric(values: list[float] | np.ndarray) -> dict[str, float]:
    """Return standard distributional summary statistics for a non-empty array."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("Cannot summarize empty array")
    q25, med, q75 = np.percentile(arr, [25.0, 50.0, 75.0])
    return {
        "min": float(np.min(arr)),
        "mean": float(np.mean(arr)),
        "median": float(med),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "iqr": float(q75 - q25),
        "q25": float(q25),
        "q75": float(q75),
        "max": float(np.max(arr)),
    }


def vargha_delaney_a12(x: np.ndarray, y: np.ndarray, *, lower_is_better: bool = True) -> float:
    """Compute exact Vargha–Delaney A_12 effect size P(X wins Y) + 0.5 * P(X == Y)."""
    if x.size == 0 or y.size == 0:
        raise ValueError("Empty sample passed to vargha_delaney_a12")
    diff = y[None, :] - x[:, None] if lower_is_better else x[:, None] - y[None, :]
    wins = float(np.sum(diff > 0.0))
    ties = float(np.sum(diff == 0.0))
    return (wins + 0.5 * ties) / float(x.size * y.size)


def cliffs_delta(x: np.ndarray, y: np.ndarray, *, lower_is_better: bool = True) -> float:
    """Compute Cliff's delta in [-1, 1] aligned so positive favors `x`."""
    return 2.0 * vargha_delaney_a12(x, y, lower_is_better=lower_is_better) - 1.0


def bootstrap_median_diff_ci(
    diff: np.ndarray,
    *,
    n_resamples: int = 10_000,
    seed: int = 20261124,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Compute reproducible percentile bootstrap CI on paired median difference."""
    if diff.size == 0:
        raise ValueError("Cannot bootstrap empty difference vector")
    rng = np.random.default_rng(seed)
    n = diff.size
    indices = rng.integers(0, n, size=(n_resamples, n))
    boot_medians = np.median(diff[indices], axis=1)
    lo = float(np.percentile(boot_medians, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(boot_medians, 100.0 * (1.0 - alpha / 2.0)))
    return lo, hi


def holm_bonferroni_adjust(p_values: list[float]) -> list[float]:
    """Apply Holm–Bonferroni step-down multiplicity correction preserving original order."""
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda kv: kv[1])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, (orig_idx, raw_p) in enumerate(indexed):
        factor = float(m - rank)
        cand = min(1.0, max(running_max, raw_p * factor))
        running_max = cand
        adjusted[orig_idx] = cand
    return adjusted


@dataclass(frozen=True, slots=True)
class FriedmanResult:
    statistic: float
    p_value: float
    kendall_w: float
    num_instances: int
    num_methods: int
    mean_ranks: dict[str, float]


def run_friedman_analysis(
    method_names: list[str],
    matrix: np.ndarray,
) -> FriedmanResult:
    """Execute instance-blocked Friedman omnibus test on shape (num_instances, num_methods)."""
    n_inst, n_meth = matrix.shape
    if n_meth != len(method_names) or n_inst < 2 or n_meth < 3:
        raise ValueError(f"Invalid Friedman matrix shape {matrix.shape} for {len(method_names)}")

    # Compute per-instance average ranks (1 = best / smallest value)
    ranks = np.empty_like(matrix, dtype=float)
    for row_idx in range(n_inst):
        row = matrix[row_idx]
        order = np.argsort(row, kind="mergesort")
        sorted_vals = row[order]
        i = 0
        while i < n_meth:
            j = i
            while j + 1 < n_meth and sorted_vals[j + 1] == sorted_vals[i]:
                j += 1
            avg_rank = 0.5 * (i + j) + 1.0
            for pos in range(i, j + 1):
                ranks[row_idx, order[pos]] = avg_rank
            i = j + 1

    mean_ranks = {name: float(np.mean(ranks[:, idx])) for idx, name in enumerate(method_names)}

    # Check if all columns identical (zero variance across every row)
    if np.all(np.ptp(matrix, axis=1) == 0.0):
        return FriedmanResult(
            statistic=0.0,
            p_value=1.0,
            kendall_w=0.0,
            num_instances=n_inst,
            num_methods=n_meth,
            mean_ranks=mean_ranks,
        )

    stat, p_val = friedmanchisquare(*[matrix[:, idx] for idx in range(n_meth)])
    chi2 = float(stat)
    kendall_w = chi2 / float(n_inst * (n_meth - 1))
    return FriedmanResult(
        statistic=chi2,
        p_value=float(p_val),
        kendall_w=float(kendall_w),
        num_instances=n_inst,
        num_methods=n_meth,
        mean_ranks=mean_ranks,
    )


def paired_comparisons_table(
    scores_by_method: dict[str, np.ndarray],
    pairs: list[tuple[str, str, str]],
    *,
    lower_is_better: bool = True,
    n_resamples: int = 10_000,
) -> list[dict[str, Any]]:
    """Run paired two-sided Wilcoxon tests + Holm-Bonferroni per family + effect sizes + CIs."""
    records: list[dict[str, Any]] = []
    by_family: dict[str, list[int]] = {}

    for idx, (family, cand, ref) in enumerate(pairs):
        x = np.asarray(scores_by_method[cand], dtype=float)
        y = np.asarray(scores_by_method[ref], dtype=float)
        diff = x - y  # negative favors cand when lower_is_better
        non_zero = diff[diff != 0.0]
        if non_zero.size == 0:
            w_stat, raw_p = 0.0, 1.0
        else:
            res = wilcoxon(x, y, alternative="two-sided", zero_method="wilcox")
            w_stat = float(res.statistic)
            raw_p = float(res.pvalue)

        ci_lo, ci_hi = bootstrap_median_diff_ci(diff, n_resamples=n_resamples)
        a12 = vargha_delaney_a12(x, y, lower_is_better=lower_is_better)
        delta = cliffs_delta(x, y, lower_is_better=lower_is_better)
        wins = int(np.sum(diff < 0.0)) if lower_is_better else int(np.sum(diff > 0.0))
        ties = int(np.sum(diff == 0.0))
        losses = int(diff.size - wins - ties)

        records.append(
            {
                "family": family,
                "candidate": cand,
                "reference": ref,
                "n_instances": int(diff.size),
                "mean_diff": float(np.mean(diff)),
                "median_diff": float(np.median(diff)),
                "ci95_median_diff_lo": ci_lo,
                "ci95_median_diff_hi": ci_hi,
                "wins_ties_losses": f"{wins}/{ties}/{losses}",
                "wilcoxon_stat": w_stat,
                "raw_p": raw_p,
                "vargha_delaney_a12": a12,
                "cliffs_delta": delta,
                "std_err_mean_diff": float(np.std(diff, ddof=1) / sqrt(diff.size))
                if diff.size > 1
                else 0.0,
            }
        )
        by_family.setdefault(family, []).append(idx)

    for row_indices in by_family.values():
        raw_ps = [records[i]["raw_p"] for i in row_indices]
        adj_ps = holm_bonferroni_adjust(raw_ps)
        for row_idx, adj_p in zip(row_indices, adj_ps, strict=True):
            records[row_idx]["holm_p"] = adj_p
            records[row_idx]["significant_0_05"] = adj_p < 0.05

    return records
