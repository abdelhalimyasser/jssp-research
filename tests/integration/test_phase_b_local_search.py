"""Integration tests for Phase B local search and metaheuristics (B01-B24, B29-B32)."""

from __future__ import annotations

from pathlib import Path

import pytest

from jssp_research.benchmarks import parse_jssp_file
from jssp_research.native import (
    default_phase_b_budget,
    generate_matched_phase_b_start,
    list_phase_b_configs,
    round_trip_machine_sequences,
    run_phase_b_batch,
    run_phase_b_single,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_ft06_jobs() -> tuple[tuple[tuple[int, int], ...], ...]:
    parsed = parse_jssp_file(
        REPO_ROOT / "tests" / "fixtures" / "medium_jssp.txt",
        instance_id="ft06",
    )
    return parsed.jobs


def test_list_28_phase_b_configs_and_pruned_d04_a_rejection() -> None:
    """Verify 28 active Phase B configs, budget helper, and D04-A rejection of B25..B28."""
    specs = list_phase_b_configs()
    assert len(specs) == 28
    codes = [s.code for s in specs]
    expected = [f"B{i:02d}" for i in range(1, 25)] + [f"B{i:02d}" for i in range(29, 33)]
    assert codes == expected
    assert default_phase_b_budget(10, 10, 100) == 10_000

    jobs = _load_ft06_jobs()
    for pruned in ("B25", "B26", "B27", "B28"):
        with pytest.raises(ValueError, match="D04-A"):
            run_phase_b_single(jobs, pruned, "ft06")


def test_matched_starts_feasibility_and_batch_reproducibility_on_ft06() -> None:
    """Verify matched starts, M2 feasibility, budget adherence, and batch parity."""
    jobs = _load_ft06_jobs()
    _, ref_sched = generate_matched_phase_b_start(jobs, initializer_id="A07", instance_id="ft06")

    tasks = [
        (code, 0, None, "paired_init", "phase_b_matched_start", code)
        for code in ("B01", "B08", "B12", "B16", "B20", "B24", "B32")
    ]
    batch_res = run_phase_b_batch(
        jobs,
        "ft06",
        tasks,
        initializer_id="A07",
        init_pop_size=64,
        evaluation_budget=500,
        num_threads=2,
    )
    assert len(batch_res) == len(tasks)

    for (code, *_), res in zip(tasks, batch_res, strict=True):
        assert res.config_id == code
        assert res.feasible_final is True
        assert res.evals_used <= 500
        assert res.initial_makespan == ref_sched.makespan
        assert res.best_makespan <= res.initial_makespan
        assert res.anytime_cmax_auc >= float(res.best_makespan)

        single = run_phase_b_single(
            jobs,
            code,
            "ft06",
            initializer_id="A07",
            init_pop_size=64,
            evaluation_budget=500,
            replicate_index=0,
        )
        assert single.best_makespan == res.best_makespan
        assert single.evals_used == res.evals_used
        assert single.best_milestones == res.best_milestones
        assert single.best_machine_sequences == res.best_machine_sequences

        rt = round_trip_machine_sequences(jobs, res.best_machine_sequences)
        assert rt.decoded_schedule.makespan <= res.best_makespan
