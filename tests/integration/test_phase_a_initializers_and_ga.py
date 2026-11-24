"""Integration tests for Phase A initializers (`A01`–`A24`) and Classical GA (`C01`)."""

from __future__ import annotations

from pathlib import Path

import pytest

from jssp_research.benchmarks import parse_jssp_file
from jssp_research.native import (
    default_phase_a_ga_budget,
    evaluate_phase_a_initializer,
    list_phase_a_initializers,
    run_phase_a_batch,
    run_phase_a_single,
    validate_obr,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_ft06_jobs() -> tuple[tuple[int, int], ...]:
    parsed = parse_jssp_file(
        REPO_ROOT / "tests" / "fixtures" / "medium_jssp.txt",
        instance_id="ft06",
    )
    return parsed.jobs


def test_list_24_phase_a_initializers_and_dimension_guard() -> None:
    """Verify all 24 Phase A initializer metadata specs and the d_max=21201 Sobol guard."""
    specs = list_phase_a_initializers()
    assert len(specs) == 24
    assert [s.code for s in specs] == [f"A{i:02d}" for i in range(1, 25)]

    by_code = {s.code: s for s in specs}
    assert by_code["A02"].is_deterministic is True
    assert by_code["A02"].alpha is None
    assert by_code["A07"].has_dyadic_sobol_net_balance is True
    assert by_code["A12"].has_dyadic_sobol_net_balance is True
    assert by_code["A14"].has_dyadic_sobol_net_balance is True
    assert by_code["A13"].has_dyadic_sobol_net_balance is False
    assert by_code["A06"].has_dyadic_sobol_net_balance is False

    # Oversized 107 x 199 (L = 21,293) instance must fail cleanly for Sobol but succeed for Halton
    row_199 = [(m, 1) for m in range(199)]
    oversized = [row_199 for _ in range(107)]
    with pytest.raises(ValueError, match="21201"):
        evaluate_phase_a_initializer(oversized, "A07", "oversized", pop_size=4)
    halton_res = evaluate_phase_a_initializer(oversized, "A04", "oversized", pop_size=4)
    assert halton_res.diagnostics.population_size == 4
    assert halton_res.diagnostics.feasible_count == 4


def test_standalone_initializer_layer_a_diagnostics_on_ft06() -> None:
    """Evaluate all 24 initializers cleanly on ft06 and check Layer A invariants."""
    jobs = _load_ft06_jobs()
    for code in [f"A{i:02d}" for i in range(1, 25)]:
        res = evaluate_phase_a_initializer(jobs, code, "ft06", pop_size=64, replicate_index=0)
        assert res.diagnostics.population_size == 64
        assert res.diagnostics.feasible_count == 64
        assert len(res.chromosomes) == 64
        assert validate_obr(jobs, res.chromosomes[0]) is True

        if code == "A02":
            assert res.diagnostics.unique_chromosomes == 1
            assert res.diagnostics.unique_schedules == 1
            assert res.diagnostics.obr_hamming_diversity == pytest.approx(0.0)
            assert res.diagnostics.qmc_1d_ks_discrepancy is None
        else:
            assert res.diagnostics.qmc_1d_ks_discrepancy is not None
            assert 0.0 < res.diagnostics.qmc_1d_ks_discrepancy < 1.0


def test_phase_a_single_and_batch_reproducibility_on_ft06() -> None:
    """Verify Python-Rust Classical GA execution, determinism, and batch parity."""
    jobs = _load_ft06_jobs()
    budget = default_phase_a_ga_budget(6, 6)
    assert budget == 6400

    single_a = run_phase_a_single(
        jobs,
        "A07",
        "ft06",
        evaluation_budget=budget,
        replicate_index=0,
        trajectory_mode="sampled",
    )
    single_b = run_phase_a_single(
        jobs,
        "A07",
        "ft06",
        evaluation_budget=budget,
        replicate_index=0,
        trajectory_mode="sampled",
    )
    assert single_a.evals_used == single_b.evals_used
    assert single_a.best_makespan == single_b.best_makespan
    assert single_a.best_found_eval == single_b.best_found_eval
    assert single_a.best_chromosome == single_b.best_chromosome
    assert single_a.best_schedule_hash == single_b.best_schedule_hash
    assert single_a.anytime_cmax_auc == pytest.approx(single_b.anytime_cmax_auc)
    assert single_a.trajectory == single_b.trajectory
    assert single_a.evals_used == budget
    assert single_a.initial.feasible_count == 64
    assert single_a.final_pop.feasible_count == 64
    assert 55 <= single_a.best_makespan <= single_a.initial.best_makespan

    tasks = [
        ("A01", 0, None, "paired_init", "phase_a_shared_init", "A01"),
        ("A07", 0, None, "paired_init", "phase_a_shared_init", "A07"),
    ]
    batch_res = run_phase_a_batch(
        jobs,
        "ft06",
        tasks,
        evaluation_budget=budget,
        num_threads=2,
        trajectory_mode="sampled",
    )
    assert len(batch_res) == 2
    assert batch_res[1].best_makespan == single_a.best_makespan
    assert batch_res[1].best_chromosome == single_a.best_chromosome
    assert batch_res[1].anytime_cmax_auc == pytest.approx(single_a.anytime_cmax_auc)
