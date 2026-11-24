"""Tests for experiment matrix builders and statistical test utilities."""

from __future__ import annotations

import numpy as np

from jssp_research.analysis import (
    holm_bonferroni_adjust,
    paired_comparisons_table,
    run_friedman_analysis,
)
from jssp_research.experiments.matrix import (
    build_phase_a_matrix,
    build_phase_b_matrix,
    build_phase_c_matrix,
    summarize_matrix_inventory,
)


def test_phase_a_and_b_and_c_matrix_inventory_counts() -> None:
    """Verify Phase A (24), Phase B (28), and Phase C (11) total exactly 63 active configs."""
    entries_a = build_phase_a_matrix()
    entries_b = build_phase_b_matrix()
    entries_c = build_phase_c_matrix()
    assert len(entries_a) == 24
    assert len(entries_b) == 28
    assert len(entries_c) == 11
    inv = summarize_matrix_inventory()
    assert inv.phase_a_count == 24
    assert inv.phase_a_ids == tuple(f"A{i:02d}" for i in range(1, 25))
    assert inv.phase_b_count == 28
    assert inv.phase_b_ids == (
        tuple(f"B{i:02d}" for i in range(1, 25)) + tuple(f"B{i:02d}" for i in range(29, 33))
    )
    assert inv.phase_c_count == 11
    assert inv.phase_c_ids == tuple(f"C{i:02d}" for i in range(1, 12))
    assert inv.total_active_configurations == 63
    assert inv.expected_total_configurations == 63


def test_friedman_and_paired_wilcoxon_holm() -> None:
    """Verify blocked Friedman rank test and Holm-Bonferroni correction on synthetic scores."""
    mat = np.array(
        [
            [1.0, 2.0, 3.0],
            [1.1, 2.1, 3.1],
            [0.9, 2.2, 3.3],
            [1.2, 2.5, 3.4],
            [1.0, 2.0, 3.5],
            [0.8, 2.1, 3.0],
        ]
    )
    fres = run_friedman_analysis(["M1", "M2", "M3"], mat)
    assert fres.mean_ranks["M1"] == 1.0
    assert fres.mean_ranks["M2"] == 2.0
    assert fres.mean_ranks["M3"] == 3.0
    assert fres.p_value < 0.01

    adj = holm_bonferroni_adjust([0.01, 0.04, 0.03])
    assert adj[0] == 0.03
    assert adj[2] == 0.06
    assert adj[1] == 0.06

    tbl = paired_comparisons_table(
        {"M1": mat[:, 0], "M2": mat[:, 1], "M3": mat[:, 2]},
        [("fam", "M1", "M2"), ("fam", "M1", "M3")],
        n_resamples=500,
    )
    assert len(tbl) == 2
    assert tbl[0]["vargha_delaney_a12"] == 1.0
    assert tbl[0]["cliffs_delta"] == 1.0
