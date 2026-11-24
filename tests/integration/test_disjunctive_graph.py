"""Python integration & property tests for M4 disjunctive graph and critical-path analysis."""

from __future__ import annotations

import json

import pytest
from hypothesis import given, seed, settings
from hypothesis import strategies as st

from jssp_research.native import (
    DisjunctiveGraphAnalysisResult,
    analyze_obr_graph,
    analyze_raw_schedule_graph,
    analyze_schedule_graph,
    build_disjunctive_graph,
    compute_head_tail_times,
    decode_obr,
    get_critical_arcs,
    get_critical_blocks,
    get_critical_machine_arcs,
    get_critical_operations,
    get_graph_makespan,
    get_representative_critical_path,
    validate_machine_sequences_dag,
)

XIE_3X3_JOBS: list[list[tuple[int, int]]] = [
    [(0, 2), (1, 1), (2, 3)],
    [(0, 1), (2, 2), (1, 2)],
    [(1, 5), (0, 2), (2, 1)],
]
XIE_3X3_CHROMOSOME: list[int] = [1, 2, 2, 2, 1, 0, 0, 0, 1]

MULTI_PATH_3X3_JOBS: list[list[tuple[int, int]]] = [
    [(0, 4), (1, 2), (2, 4)],
    [(1, 4), (2, 1), (0, 1)],
    [(2, 2), (0, 3), (1, 1)],
]
MULTI_PATH_3X3_CHROMOSOME: list[int] = [0, 1, 2, 1, 1, 0, 0, 2, 2]

TWO_BLOCKS_4X3_JOBS: list[list[tuple[int, int]]] = [
    [(0, 2), (1, 1), (2, 1)],
    [(0, 3), (1, 6), (2, 2)],
    [(2, 8), (0, 2), (1, 1)],
    [(1, 1), (2, 1), (0, 3)],
]
TWO_BLOCKS_4X3_CHROMOSOME: list[int] = [0, 1, 2, 0, 0, 1, 1, 3, 3, 2, 3, 2]


def test_xie_3x3_end_to_end_disjunctive_graph_critical_path_and_blocks() -> None:
    """Verify complete M4 analysis on the documented 3x3 Xie et al. (2023) fixture."""
    schedule = decode_obr(XIE_3X3_JOBS, XIE_3X3_CHROMOSOME)
    built = build_disjunctive_graph(XIE_3X3_JOBS, schedule)
    analyzed = analyze_schedule_graph(XIE_3X3_JOBS, schedule)
    via_obr = analyze_obr_graph(XIE_3X3_JOBS, XIE_3X3_CHROMOSOME)

    assert isinstance(analyzed, DisjunctiveGraphAnalysisResult)
    assert built == analyzed == via_obr
    assert get_graph_makespan(XIE_3X3_JOBS, schedule) == 11
    assert analyzed.schedule_makespan == 11

    assert analyzed.metadata.num_jobs == 3
    assert analyzed.metadata.num_machines == 3
    assert analyzed.metadata.num_operations == 9
    assert analyzed.metadata.num_vertices == 11
    assert analyzed.metadata.num_job_precedence_arcs == 6
    assert analyzed.metadata.num_machine_order_arcs == 6
    assert analyzed.metadata.num_source_arcs == 2
    assert analyzed.metadata.num_sink_arcs == 2
    assert analyzed.metadata.num_arcs == 16

    ht = {
        (rec.job, rec.op_index): (rec.head, rec.tail, rec.slack, rec.is_critical)
        for rec in compute_head_tail_times(XIE_3X3_JOBS, schedule)
    }
    assert ht[(2, 0)] == (0, 6, 0, True)
    assert ht[(2, 1)] == (5, 4, 0, True)
    assert ht[(2, 2)] == (7, 3, 0, True)
    assert ht[(0, 2)] == (8, 0, 0, True)
    assert ht[(0, 0)] == (1, 6, 2, False)
    assert ht[(1, 1)] == (1, 4, 4, False)

    assert get_critical_operations(XIE_3X3_JOBS, schedule) == (
        (0, 2),
        (2, 0),
        (2, 1),
        (2, 2),
    )
    rep = get_representative_critical_path(XIE_3X3_JOBS, schedule)
    assert rep.operations == ((2, 0), (2, 1), (2, 2), (0, 2))
    assert rep.total_processing_time == 11

    crit_arcs = get_critical_arcs(XIE_3X3_JOBS, schedule)
    assert len(crit_arcs) == 5
    crit_machine_arcs = get_critical_machine_arcs(XIE_3X3_JOBS, schedule)
    assert len(crit_machine_arcs) == 1
    assert crit_machine_arcs[0].from_op == (2, 2)
    assert crit_machine_arcs[0].to_op == (0, 2)
    assert crit_machine_arcs[0].owner_id == 2

    blocks = get_critical_blocks(XIE_3X3_JOBS, schedule)
    assert len(blocks) == 1
    assert blocks[0].machine == 2
    assert blocks[0].operations == ((2, 2), (0, 2))
    assert blocks[0].length == 2

    # Verify JSON serializability via to_dict()
    serialized = json.dumps(analyzed.to_dict(), sort_keys=True)
    assert '"graph_makespan": 11' in serialized


def test_multi_path_critical_ops_and_multiple_blocks_on_one_machine() -> None:
    """Verify critical ops off representative path, non-critical machine arcs, and blocks."""
    mp_res = analyze_obr_graph(MULTI_PATH_3X3_JOBS, MULTI_PATH_3X3_CHROMOSOME)
    assert mp_res.graph_makespan == 10
    assert mp_res.representative_critical_path.operations == ((0, 0), (0, 1), (0, 2))
    assert (1, 1) in mp_res.critical_operations
    assert (2, 1) in mp_res.critical_operations
    assert mp_res.operation(0, 1).maximizing_predecessors == ((0, 0), (1, 0))
    assert mp_res.operation(1, 0).maximizing_successors == ((0, 1), (1, 1))

    # Non-critical machine arcs between critical vertices are excluded from critical blocks
    for arc in mp_res.arcs:
        if (arc.from_op, arc.to_op) in {
            ((0, 0), (1, 2)),
            ((1, 1), (0, 2)),
            ((0, 1), (2, 2)),
        }:
            assert arc.arc_type == "machine_order"
            assert not arc.is_head_tight
            assert not arc.is_critical

    assert len(mp_res.critical_blocks) == 2
    assert mp_res.critical_blocks[0].operations == ((1, 2), (2, 1))
    assert mp_res.critical_blocks[1].operations == ((1, 0), (0, 1))

    # Multiple blocks on M0 plus 3-op block on M2 in 4x3 fixture
    blocks_4x3 = analyze_obr_graph(TWO_BLOCKS_4X3_JOBS, TWO_BLOCKS_4X3_CHROMOSOME)
    assert blocks_4x3.graph_makespan == 13
    assert len(blocks_4x3.critical_blocks) == 3
    assert blocks_4x3.critical_blocks[0].machine == 0
    assert blocks_4x3.critical_blocks[0].operations == ((0, 0), (1, 0))
    assert blocks_4x3.critical_blocks[1].machine == 0
    assert blocks_4x3.critical_blocks[1].operations == ((2, 1), (3, 2))
    assert blocks_4x3.critical_blocks[2].machine == 2
    assert blocks_4x3.critical_blocks[2].operations == ((2, 0), (0, 2), (3, 1))


def test_graph_cycle_and_error_translation_over_ffi() -> None:
    """Verify cycle, right-shift mismatch, and overflow error mapping across PyO3."""
    jobs_2x2 = [[(0, 2), (1, 3)], [(1, 4), (0, 1)]]

    # Valid acyclic sequences pass DAG check
    topo = validate_machine_sequences_dag(
        jobs_2x2,
        [[(0, 0), (1, 1)], [(1, 0), (0, 1)]],
    )
    assert topo[0].kind == "source"
    assert topo[-1].kind == "sink"

    # Reversed sequences induce cycle O(0,0)->O(0,1)->O(1,0)->O(1,1)->O(0,0)
    with pytest.raises(ValueError, match="directed cycle"):
        validate_machine_sequences_dag(
            jobs_2x2,
            [[(1, 1), (0, 0)], [(0, 1), (1, 0)]],
        )

    # Artificial right-shifted schedule fails GraphMakespanMismatch
    with pytest.raises(ValueError, match="disagrees with schedule makespan"):
        analyze_raw_schedule_graph(
            [[(0, 5)]],
            [(0, 0, 0, 2, 7, 5)],
            7,
        )

    # u64 overflow translates cleanly to OverflowError
    with pytest.raises(OverflowError, match="overflowed"):
        analyze_obr_graph([[(0, (1 << 64) - 1), (1, 1)], [(0, 1), (1, 1)]], [0, 1, 0, 1])


@st.composite
def valid_rectangular_instance_and_obr(
    draw: st.DrawFn,
) -> tuple[list[list[tuple[int, int]]], list[int]]:
    """Generate small random rectangular JSSP instances and valid OBR strings."""
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


@seed(20261124)
@settings(max_examples=35, deadline=None)
@given(data=valid_rectangular_instance_and_obr())
def test_hypothesis_bridge_disjunctive_graph_properties(
    data: tuple[list[list[tuple[int, int]]], list[int]],
) -> None:
    """Property test verifying graph, head/tail, critical-path, and block invariants in Python."""
    jobs, chromosome = data
    schedule = decode_obr(jobs, chromosome)
    analysis = analyze_schedule_graph(jobs, schedule)
    analysis_repeat = analyze_schedule_graph(jobs, schedule)

    assert analysis == analysis_repeat
    assert analysis.graph_makespan == schedule.makespan
    assert analysis.representative_critical_path.total_processing_time == analysis.graph_makespan
    assert len(analysis.critical_operations) >= 1

    # Every op satisfies head == schedule start_time and non-negative slack
    for op_rec in analysis.operations:
        sched_op = schedule.operation(op_rec.job, op_rec.op_index)
        assert op_rec.head == sched_op.start_time
        assert op_rec.head + op_rec.processing_time + op_rec.tail <= analysis.graph_makespan
        assert op_rec.slack == analysis.graph_makespan - (
            op_rec.head + op_rec.processing_time + op_rec.tail
        )
        assert op_rec.is_critical == (op_rec.slack == 0)

    # Critical block consistency and maximality
    for block in analysis.critical_blocks:
        assert block.length >= 2
        assert len(block.operations) == block.length
        assert len(block.critical_machine_arc_ids) == block.length - 1
        m_seq = schedule.machine_sequences[block.machine]
        start_idx = m_seq.index(block.first_operation)
        assert m_seq[start_idx : start_idx + block.length] == block.operations
        for arc_id in block.critical_machine_arc_ids:
            arc = analysis.arcs[arc_id]
            assert arc.arc_type == "machine_order"
            assert arc.is_critical
