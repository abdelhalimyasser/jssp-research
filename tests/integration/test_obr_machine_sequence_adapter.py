"""Python integration & Hypothesis tests for M5 OBR <-> Machine-Sequence adapter."""

from __future__ import annotations

import json

import pytest
from hypothesis import given, seed, settings
from hypothesis import strategies as st

from jssp_research.native import (
    AdapterAuditReportRecord,
    AdapterRoundTripResult,
    EncodedObrResultRecord,
    build_machine_sequence_graph,
    decode_obr,
    decode_obr_append_only,
    encode_machine_sequences_to_obr,
    round_trip_machine_sequences,
    round_trip_obr_schedule,
    validate_schedule,
)

XIE_3X3_JOBS: list[list[tuple[int, int]]] = [
    [(0, 2), (1, 1), (2, 3)],
    [(0, 1), (2, 2), (1, 2)],
    [(1, 5), (0, 2), (2, 1)],
]
XIE_3X3_CHROMOSOME: list[int] = [1, 2, 2, 2, 1, 0, 0, 0, 1]

# Section 6 adversarial 2x2 gap-insertion fixture
ADVERSARIAL_2X2_JOBS: list[list[tuple[int, int]]] = [
    [(1, 5), (0, 4)],
    [(0, 2), (1, 3)],
]
ADVERSARIAL_2X2_TARGET_SEQS: list[list[tuple[int, int]]] = [
    [(0, 1), (1, 0)],
    [(0, 0), (1, 1)],
]


def test_xie_3x3_adapter_exact_round_trip_over_ffi() -> None:
    """Verify exact D01-B round-trip for Xie 3x3 M2 active schedule over PyO3."""
    schedule = decode_obr(XIE_3X3_JOBS, XIE_3X3_CHROMOSOME)
    graph_from_seqs = build_machine_sequence_graph(XIE_3X3_JOBS, schedule.machine_sequences)
    assert graph_from_seqs.graph_makespan == 11

    encoded = encode_machine_sequences_to_obr(XIE_3X3_JOBS, schedule.machine_sequences)
    assert isinstance(encoded, EncodedObrResultRecord)
    assert encoded.chromosome == (1, 2, 0, 1, 0, 2, 1, 2, 0)
    assert encoded.obr_valid is True
    assert encoded.technological_order_preserved is True
    assert encoded.job_multiplicities == (3, 3, 3)

    rt_from_seqs = round_trip_machine_sequences(XIE_3X3_JOBS, schedule.machine_sequences)
    rt_from_obr = round_trip_obr_schedule(XIE_3X3_JOBS, XIE_3X3_CHROMOSOME)
    assert isinstance(rt_from_seqs, AdapterRoundTripResult)
    assert rt_from_seqs == rt_from_obr
    assert isinstance(rt_from_seqs.audit, AdapterAuditReportRecord)
    assert rt_from_seqs.audit.is_exact is True
    assert rt_from_seqs.audit.status == "exact"
    assert rt_from_seqs.audit.target_graph_makespan == 11
    assert rt_from_seqs.audit.decoded_schedule_makespan == 11
    assert rt_from_seqs.audit.signed_makespan_diff == 0
    assert rt_from_seqs.audit.adapter_left_shift_count == 0
    assert rt_from_seqs.audit.adapter_right_shift_count == 0
    assert rt_from_seqs.audit.adapter_unchanged_start_count == 9
    assert rt_from_seqs.audit.machine_order_inversion_count == 0
    assert rt_from_seqs.audit.mismatched_machine_arc_count == 0

    serialized = json.dumps(rt_from_seqs.audit.to_dict(), sort_keys=True)
    assert '"status": "exact"' in serialized


def test_section_6_adversarial_2x2_gap_insertion_over_ffi() -> None:
    """Verify the Section 6 2x2 adversarial fixture shows gap-insertion order change."""
    target_graph = build_machine_sequence_graph(ADVERSARIAL_2X2_JOBS, ADVERSARIAL_2X2_TARGET_SEQS)
    assert target_graph.graph_makespan == 14
    assert target_graph.operation(0, 0).head == 0
    assert target_graph.operation(0, 1).head == 5
    assert target_graph.operation(1, 0).head == 9
    assert target_graph.operation(1, 1).head == 11

    rt = round_trip_machine_sequences(ADVERSARIAL_2X2_JOBS, ADVERSARIAL_2X2_TARGET_SEQS)
    assert rt.encoded.chromosome == (0, 0, 1, 1)
    assert rt.encoded.obr_valid is True
    assert rt.encoded.technological_order_preserved is True
    assert validate_schedule(ADVERSARIAL_2X2_JOBS, rt.decoded_schedule) is True

    assert rt.decoded_schedule.makespan == 9
    assert rt.decoded_schedule.machine_sequences == (
        ((1, 0), (0, 1)),
        ((0, 0), (1, 1)),
    )
    assert rt.audit.status == "machine_order_and_start_times_changed"
    assert rt.audit.is_exact is False
    assert rt.audit.signed_makespan_diff == -5
    assert rt.audit.adapter_left_shift_count == 2
    assert rt.audit.adapter_right_shift_count == 0
    assert rt.audit.adapter_unchanged_start_count == 2
    assert rt.audit.machine_order_inversion_count == 1
    assert rt.audit.mismatched_machine_arc_count == 1


def test_adapter_error_translation_for_malformed_and_cyclic_machine_sequences() -> None:
    """Verify invalid, duplicate, wrong-machine, missing, and cyclic errors map to ValueError."""
    # Cyclic 2x2 target sequences
    with pytest.raises(ValueError, match="directed cycle"):
        round_trip_machine_sequences(
            ADVERSARIAL_2X2_JOBS,
            [[(0, 1), (1, 0)], [(1, 1), (0, 0)]],
        )

    # Wrong machine assignment
    with pytest.raises(ValueError, match="assigned to machine"):
        build_machine_sequence_graph(
            ADVERSARIAL_2X2_JOBS,
            [[(0, 0), (1, 0)], [(0, 1), (1, 1)]],
        )

    # Duplicate operation
    with pytest.raises(ValueError, match="duplicate operation"):
        encode_machine_sequences_to_obr(
            ADVERSARIAL_2X2_JOBS,
            [[(1, 0), (1, 0)], [(0, 0), (1, 1)]],
        )

    # Truncated sequence (missing operation)
    with pytest.raises(ValueError, match="expected 2"):
        round_trip_machine_sequences(
            ADVERSARIAL_2X2_JOBS,
            [[(0, 1)], [(0, 0), (1, 1)]],
        )


@st.composite
def valid_rectangular_instance_and_obr(
    draw: st.DrawFn,
) -> tuple[list[list[tuple[int, int]]], list[int]]:
    """Generate small random rectangular JSSP instances and valid OBR chromosomes."""
    num_jobs = draw(st.integers(min_value=1, max_value=5))
    num_machines = draw(st.integers(min_value=1, max_value=5))
    jobs: list[list[tuple[int, int]]] = []
    for _ in range(num_jobs):
        machine_perm = draw(st.permutations(list(range(num_machines))))
        durations = draw(
            st.lists(
                st.integers(min_value=1, max_value=25),
                min_size=num_machines,
                max_size=num_machines,
            )
        )
        jobs.append(list(zip(machine_perm, durations, strict=True)))

    base_genes = [j for j in range(num_jobs) for _ in range(num_machines)]
    chromosome = list(draw(st.permutations(base_genes)))
    return jobs, chromosome


@seed(2026112405)
@settings(max_examples=35, deadline=None)
@given(data=valid_rectangular_instance_and_obr())
def test_hypothesis_adapter_active_exactness_and_semi_active_bounds(
    data: tuple[list[list[tuple[int, int]]], list[int]],
) -> None:
    """Verify Category A (exact) and Category B (bounded left-shift) properties over FFI."""
    jobs, chromosome = data

    # Category A: M2 Serial SGS active schedules always round-trip with exact status
    active_sched = decode_obr(jobs, chromosome)
    rt_active = round_trip_obr_schedule(jobs, chromosome)
    rt_active_repeat = round_trip_machine_sequences(jobs, active_sched.machine_sequences)

    assert rt_active == rt_active_repeat
    assert rt_active.encoded.obr_valid is True
    assert rt_active.encoded.technological_order_preserved is True
    assert rt_active.audit.status == "exact"
    assert rt_active.audit.adapter_left_shift_count == 0
    assert rt_active.audit.adapter_right_shift_count == 0
    assert rt_active.audit.machine_order_inversion_count == 0
    assert rt_active.audit.signed_makespan_diff == 0

    # Category B: Append-only semi-active machine sequences always yield valid OBR & <= makespan
    semi_active = decode_obr_append_only(jobs, chromosome)
    rt_semi = round_trip_machine_sequences(jobs, semi_active.machine_sequences)
    assert rt_semi.encoded.obr_valid is True
    assert rt_semi.encoded.technological_order_preserved is True
    assert validate_schedule(jobs, rt_semi.decoded_schedule) is True
    assert rt_semi.audit.adapter_right_shift_count == 0
    assert rt_semi.audit.signed_makespan_diff <= 0
