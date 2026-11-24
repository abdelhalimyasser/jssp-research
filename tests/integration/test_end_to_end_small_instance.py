"""End-to-end Python-Rust tests for OBR decoding, feasibility validation, and properties."""

from __future__ import annotations

import pytest
from hypothesis import given, seed, settings
from hypothesis import strategies as st

from jssp_research.native import (
    DecodedScheduleResult,
    decode_obr,
    decode_obr_append_only,
    evaluate_makespan,
    resolve_obr_operations,
    validate_raw_schedule,
    validate_schedule,
)

XIE_3X3_JOBS: list[list[tuple[int, int]]] = [
    [(0, 2), (1, 1), (2, 3)],
    [(0, 1), (2, 2), (1, 2)],
    [(1, 5), (0, 2), (2, 1)],
]
XIE_3X3_CHROMOSOME: list[int] = [1, 2, 2, 2, 1, 0, 0, 0, 1]


def test_xie_3x3_end_to_end_gap_insertion_vs_append_only() -> None:
    """Verify exact 3x3 benchmark makespan (11 vs 13), intervals, and sequences via FFI."""
    resolved = resolve_obr_operations(XIE_3X3_JOBS, XIE_3X3_CHROMOSOME)
    assert resolved == [
        (1, 0),
        (2, 0),
        (2, 1),
        (2, 2),
        (1, 1),
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 2),
    ]

    schedule = decode_obr(XIE_3X3_JOBS, XIE_3X3_CHROMOSOME)
    assert isinstance(schedule, DecodedScheduleResult)
    assert schedule.makespan == 11
    assert validate_schedule(XIE_3X3_JOBS, schedule) is True
    assert evaluate_makespan(XIE_3X3_JOBS, schedule) == 11

    assert schedule.operation(1, 1).start_time == 1
    assert schedule.operation(1, 1).completion_time == 3
    assert schedule.operation(0, 0).start_time == 1
    assert schedule.operation(0, 0).completion_time == 3
    assert schedule.operation(0, 2).start_time == 8
    assert schedule.operation(0, 2).completion_time == 11

    assert schedule.machine_sequences == (
        ((1, 0), (0, 0), (2, 1)),
        ((2, 0), (0, 1), (1, 2)),
        ((1, 1), (2, 2), (0, 2)),
    )

    append_schedule = decode_obr_append_only(XIE_3X3_JOBS, XIE_3X3_CHROMOSOME)
    assert append_schedule.makespan == 13
    assert append_schedule.operation(1, 1).start_time == 8
    assert append_schedule.operation(1, 2).completion_time == 12
    assert append_schedule.operation(0, 2).completion_time == 13


def test_python_feasibility_catches_corrupted_schedules() -> None:
    """Verify `validate_raw_schedule` rejects precedence, overlap, and makespan errors."""
    schedule = decode_obr(XIE_3X3_JOBS, XIE_3X3_CHROMOSOME)
    raw_ops = schedule.to_raw_operations()

    # Corrupt reported makespan
    with pytest.raises(ValueError, match="recomputed maximum completion time"):
        validate_raw_schedule(XIE_3X3_JOBS, raw_ops, schedule.makespan + 1)

    # Corrupt completion time of O(0, 0) (index 0)
    bad_duration = list(raw_ops)
    j, k, m, s, c, p = bad_duration[0]
    bad_duration[0] = (j, k, m, s, c + 1, p)
    with pytest.raises(ValueError, match="does not match processing time"):
        validate_raw_schedule(XIE_3X3_JOBS, bad_duration, schedule.makespan)


@st.composite
def valid_rectangular_instance_and_obr(
    draw: st.DrawFn,
) -> tuple[list[list[tuple[int, int]]], list[int]]:
    """Generate small random rectangular classical JSSP instances and valid OBR strings."""
    num_jobs = draw(st.integers(min_value=1, max_value=5))
    num_machines = draw(st.integers(min_value=1, max_value=5))

    jobs: list[list[tuple[int, int]]] = []
    for _ in range(num_jobs):
        machine_perm = draw(st.permutations(list(range(num_machines))))
        durations = draw(
            st.lists(
                st.integers(min_value=1, max_value=30),
                min_size=num_machines,
                max_size=num_machines,
            )
        )
        jobs.append(list(zip(machine_perm, durations, strict=True)))

    base_genes = [j for j in range(num_jobs) for _ in range(num_machines)]
    chromosome = list(draw(st.permutations(base_genes)))
    return jobs, chromosome


@seed(20260417)
@settings(max_examples=40, deadline=None)
@given(data=valid_rectangular_instance_and_obr())
def test_hypothesis_bridge_decoder_properties(
    data: tuple[list[list[tuple[int, int]]], list[int]],
) -> None:
    """Property test verifying all M2 decoding invariants across the Python-Rust FFI."""
    jobs, chromosome = data
    num_jobs = len(jobs)
    num_machines = len(jobs[0])

    schedule = decode_obr(jobs, chromosome)
    schedule_repeat = decode_obr(jobs, chromosome)
    append_schedule = decode_obr_append_only(jobs, chromosome)

    # Exact operation count, independent validation, and makespan agreement
    assert len(schedule.operations) == num_jobs * num_machines
    assert validate_schedule(jobs, schedule) is True
    assert validate_schedule(jobs, append_schedule) is True
    assert evaluate_makespan(jobs, schedule) == schedule.makespan
    assert schedule.makespan <= append_schedule.makespan
    assert schedule == schedule_repeat

    # Check precedence and interval consistency directly in Python
    op_map = {(op.job, op.op_index): op for op in schedule.operations}
    for j in range(num_jobs):
        for k in range(num_machines):
            cur = op_map[(j, k)]
            assert cur.start_time + cur.processing_time == cur.completion_time
            if k > 0:
                prev = op_map[(j, k - 1)]
                assert prev.completion_time <= cur.start_time

    # Check machine exclusivity across machine sequences
    for m_seq in schedule.machine_sequences:
        assert len(m_seq) == num_jobs
        intervals = [(op_map[op_id].start_time, op_map[op_id].completion_time) for op_id in m_seq]
        for idx in range(len(intervals) - 1):
            assert intervals[idx][1] <= intervals[idx + 1][0]
