"""Integration tests for Phase C Memetic & Dual-View GAs (C01-C11) and Integrated P5."""

from __future__ import annotations

from pathlib import Path

from jssp_research.benchmarks import parse_jssp_file
from jssp_research.native import (
    decode_obr,
    default_phase_c_budget,
    list_phase_c_configs,
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
