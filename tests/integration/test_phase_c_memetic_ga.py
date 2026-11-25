"""Integration tests for Phase C Memetic & Dual-View GAs (C01-C11) and Integrated P5."""

from __future__ import annotations

from pathlib import Path

from jssp_research.benchmarks import parse_jssp_file
from jssp_research.native import (
    decode_obr,
    default_phase_c_budget,
    list_phase_c_configs,
    run_dual_view_exploratory_batch,
    run_dual_view_exploratory_single,
    run_integrated_acb_batch,
    run_phase_c_batch,
    run_phase_c_single,
    validate_schedule,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_ft06_jobs() -> tuple[tuple[tuple[int, int], ...], ...]:
    parsed = parse_jssp_file(
        REPO_ROOT / "tests" / "fixtures" / "medium_jssp.txt",
        instance_id="ft06",
    )
    return parsed.jobs


def test_list_11_phase_c_configs_and_run_batch_on_ft06() -> None:
    """Verify all 11 Phase C configs, equalized budgets, M2 feasibility, and P5 pipeline."""
    specs = list_phase_c_configs()
    assert len(specs) == 11
    assert [s.code for s in specs] == [f"C{i:02d}" for i in range(1, 12)]
    assert default_phase_c_budget(6, 6) == 6400

    jobs = _load_ft06_jobs()
    tasks = [(s.code, 0, None, "paired_init", "phase_c_shared_init", s.code) for s in specs]
    batch = run_phase_c_batch(
        jobs,
        "ft06",
        tasks,
        initializer_id="A07",
        ls_neighborhood="N7",
        ls_strategy="steepest_descent",
        evaluation_budget=800,
        num_threads=2,
    )
    assert len(batch) == 11

    for spec, res in zip(specs, batch, strict=True):
        assert res.config_id == spec.code
        assert res.genotype_capacity == 64
        assert res.feasible_final is True
        assert res.evals_used <= 800
        assert res.decode_evals + res.ls_evals == res.evals_used
        sched = decode_obr(jobs, res.best_chromosome)
        assert sched.makespan == res.best_makespan
        assert validate_schedule(jobs, sched) is True

        single = run_phase_c_single(
            jobs,
            spec.code,
            "ft06",
            initializer_id="A07",
            ls_neighborhood="N7",
            ls_strategy="steepest_descent",
            evaluation_budget=800,
            replicate_index=0,
        )
        assert single.best_makespan == res.best_makespan
        assert single.evals_used == res.evals_used
        assert single.best_chromosome == res.best_chromosome

    # Integrated P5 pipeline check
    p5_res = run_integrated_acb_batch(
        jobs,
        "ft06",
        [0, 1],
        frozen_a_id="A07",
        frozen_b_id="B03",
        frozen_c_id="C06",
        evaluation_budget=1000,
        num_threads=2,
    )
    assert len(p5_res) == 2
    for row in p5_res:
        assert row.feasible_final is True
        assert row.evals_used <= 1000
        assert row.final_best_makespan <= row.stage_c_best_makespan


def test_exploratory_dual_view_ls_budget_allocation_and_parity() -> None:
    """Verify exploratory Dual-View LS budget parity with C06, conservation, and strategies."""
    jobs = _load_ft06_jobs()
    budget = 900

    c06 = run_phase_c_single(
        jobs,
        "C06",
        "ft06",
        initializer_id="A11",
        ls_neighborhood="N5",
        ls_strategy="steepest_descent",
        max_ls_evals=50,
        evaluation_budget=budget,
        replicate_index=0,
        configuration_id="C06",
    )
    ctrl = run_dual_view_exploratory_single(
        jobs,
        "DV_A4_B50",
        "ft06",
        base_config_id="C06",
        initializer_id="A11",
        ls_neighborhood="N5",
        ls_strategy="steepest_descent",
        max_ls_evals=50,
        candidate_fraction=1.0,
        evaluation_budget=budget,
        replicate_index=0,
        configuration_id="C06",
    )
    assert ctrl.best_makespan == c06.best_makespan
    assert ctrl.evals_used == c06.evals_used
    assert ctrl.generations_executed == c06.generations_executed
    assert ctrl.final_gate_p == c06.final_gate_p
    assert ctrl.best_chromosome == c06.best_chromosome

    tasks = [
        (
            "DV_A1_B10",
            "C06",
            "N5",
            "steepest_descent",
            10,
            1.0,
            1,
            None,
            "paired_init",
            "phase_c_shared_init",
            "C06",
        ),
        (
            "DV_B1_F020",
            "C06",
            "N5",
            "steepest_descent",
            20,
            0.20,
            1,
            None,
            "paired_init",
            "phase_c_shared_init",
            "C06",
        ),
        (
            "DV_C2_FI",
            "C06",
            "N5",
            "first_improvement",
            20,
            0.20,
            1,
            None,
            "paired_init",
            "phase_c_shared_init",
            "C06",
        ),
        (
            "DV_C3_SA",
            "C06",
            "N5",
            "budget_limited_sa",
            20,
            0.20,
            1,
            None,
            "paired_init",
            "phase_c_shared_init",
            "C06",
        ),
    ]
    batch = run_dual_view_exploratory_batch(
        jobs,
        "ft06",
        tasks,
        initializer_id="A11",
        evaluation_budget=budget,
        num_threads=2,
    )
    assert len(batch) == 4
    for r in batch:
        assert r.feasible_final is True
        assert r.evals_used <= budget
        assert r.init_evals + r.search_evals == r.evals_used
        branch_evals = (
            r.lam_branch.decode_evals
            + r.lam_branch.ls_evals
            + r.bald_branch.decode_evals
            + r.bald_branch.ls_evals
        )
        assert branch_evals == r.evals_used
        assert len(r.diversity_trajectory) == r.generations_executed + 1
        sched = decode_obr(jobs, r.best_chromosome)
        assert sched.makespan == r.best_makespan
        assert validate_schedule(jobs, sched) is True

    # Selective LS runs more generations than 100% LS under equal budget
    assert batch[1].ls_strategy == "steepest_descent"
    assert batch[1].lam_branch.ls_calls < batch[1].lam_branch.decode_evals
    assert batch[3].ls_strategy == "budget_limited_sa"
    assert batch[3].lam_branch.accepted_moves > 0
