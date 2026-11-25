"""Thin typed Python wrapper around the compiled `jssp_research._native` extension."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from jssp_research import _native

type InstanceSpec = Sequence[Sequence[tuple[int, int]]]
type RawScheduledOperationTuple = tuple[int, int, int, int, int, int]


@dataclass(frozen=True, slots=True)
class ScheduledOperationRecord:
    """Immutable Python record mirroring a Rust `ScheduledOperation`."""

    job: int
    op_index: int
    machine: int
    start_time: int
    completion_time: int
    processing_time: int


@dataclass(frozen=True, slots=True)
class RngMetadataRecord:
    """Immutable snapshot of Rust portable PRNG algorithm and seed-derivation constants."""

    algorithm: str
    rng_crate_version: str
    hash_crate_version: str
    chacha_rounds: int
    seed_derivation_schema: str
    master_seed_schedule_id: str
    stream_roles: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DerivedStreamPreview:
    """Derived 256-bit seed metadata and preview draws for one named RNG stream."""

    role: str
    coupling_key: str
    seed_hex: str
    preview_draws_u64: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class DecodedScheduleResult:
    """Immutable Python container for a decoded and validated classical JSSP schedule."""

    makespan: int
    operations: tuple[ScheduledOperationRecord, ...]
    machine_sequences: tuple[tuple[tuple[int, int], ...], ...]

    def operation(self, job: int, op_index: int) -> ScheduledOperationRecord:
        """Return the scheduled operation record matching `(job, op_index)`."""
        for op in self.operations:
            if op.job == job and op.op_index == op_index:
                return op
        raise KeyError(f"Operation ({job}, {op_index}) not found in schedule")

    def to_raw_operations(self) -> list[RawScheduledOperationTuple]:
        """Serialize scheduled operations back into native FFI tuples."""
        return [
            (
                op.job,
                op.op_index,
                op.machine,
                op.start_time,
                op.completion_time,
                op.processing_time,
            )
            for op in self.operations
        ]


@dataclass(frozen=True, slots=True)
class GraphMetadataRecord:
    """Summary counts for a constructed compact oriented disjunctive graph."""

    num_jobs: int
    num_machines: int
    num_operations: int
    num_vertices: int
    num_arcs: int
    num_job_precedence_arcs: int
    num_machine_order_arcs: int
    num_source_arcs: int
    num_sink_arcs: int


@dataclass(frozen=True, slots=True)
class GraphVertexRecord:
    """Vertex entry in deterministic topological order."""

    vertex_id: int
    kind: str
    op_id: tuple[int, int] | None


@dataclass(frozen=True, slots=True)
class GraphArcRecord:
    """Immutable Python record mirroring a classified Rust `AnalyzedArc`."""

    arc_id: int
    from_vertex: int
    to_vertex: int
    from_op: tuple[int, int] | None
    to_op: tuple[int, int] | None
    arc_type: str
    owner_id: int | None
    is_head_tight: bool
    is_critical: bool


@dataclass(frozen=True, slots=True)
class OperationCriticalityRecord:
    """Immutable Python record mirroring a Rust `OperationCriticality` entry."""

    job: int
    op_index: int
    vertex_id: int
    machine: int
    processing_time: int
    head: int
    tail: int
    slack: int
    is_critical: bool
    maximizing_predecessors: tuple[tuple[int, int], ...]
    representative_predecessor: tuple[int, int] | None
    maximizing_successors: tuple[tuple[int, int], ...]
    representative_successor: tuple[int, int] | None


@dataclass(frozen=True, slots=True)
class CriticalBlockRecord:
    """Immutable Python record mirroring a maximal extracted `CriticalBlock`."""

    block_id: int
    machine: int
    operations: tuple[tuple[int, int], ...]
    first_operation: tuple[int, int]
    last_operation: tuple[int, int]
    length: int
    critical_machine_arc_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class RepresentativeCriticalPathRecord:
    """Deterministic representative Source-to-Sink critical path from Rust."""

    operations: tuple[tuple[int, int], ...]
    arc_ids: tuple[int, ...]
    total_processing_time: int


@dataclass(frozen=True, slots=True)
class DisjunctiveGraphAnalysisResult:
    """Immutable Python container holding Rust disjunctive-graph & critical-path analysis."""

    graph_makespan: int
    schedule_makespan: int
    metadata: GraphMetadataRecord
    topological_order: tuple[GraphVertexRecord, ...]
    operations: tuple[OperationCriticalityRecord, ...]
    arcs: tuple[GraphArcRecord, ...]
    critical_operations: tuple[tuple[int, int], ...]
    critical_arc_ids: tuple[int, ...]
    critical_machine_arc_ids: tuple[int, ...]
    critical_job_arc_ids: tuple[int, ...]
    representative_critical_path: RepresentativeCriticalPathRecord
    critical_blocks: tuple[CriticalBlockRecord, ...]

    def operation(self, job: int, op_index: int) -> OperationCriticalityRecord:
        """Return the `OperationCriticalityRecord` matching `(job, op_index)`."""
        for op in self.operations:
            if op.job == job and op.op_index == op_index:
                return op
        raise KeyError(f"Operation ({job}, {op_index}) not found in graph analysis")

    @property
    def critical_arcs(self) -> tuple[GraphArcRecord, ...]:
        """Return all classified critical arcs in canonical arc order."""
        return tuple(self.arcs[arc_id] for arc_id in self.critical_arc_ids)

    @property
    def critical_machine_arcs(self) -> tuple[GraphArcRecord, ...]:
        """Return all critical `machine_order` arcs in canonical arc order."""
        return tuple(self.arcs[arc_id] for arc_id in self.critical_machine_arc_ids)

    @property
    def critical_job_arcs(self) -> tuple[GraphArcRecord, ...]:
        """Return all critical `job_precedence` arcs in canonical arc order."""
        return tuple(self.arcs[arc_id] for arc_id in self.critical_job_arc_ids)

    def to_dict(self) -> dict[str, object]:
        """Convert the complete graph analysis to deterministic JSON-serializable primitives."""
        return {
            "graph_makespan": self.graph_makespan,
            "schedule_makespan": self.schedule_makespan,
            "metadata": {
                "num_jobs": self.metadata.num_jobs,
                "num_machines": self.metadata.num_machines,
                "num_operations": self.metadata.num_operations,
                "num_vertices": self.metadata.num_vertices,
                "num_arcs": self.metadata.num_arcs,
                "num_job_precedence_arcs": self.metadata.num_job_precedence_arcs,
                "num_machine_order_arcs": self.metadata.num_machine_order_arcs,
                "num_source_arcs": self.metadata.num_source_arcs,
                "num_sink_arcs": self.metadata.num_sink_arcs,
            },
            "topological_order": [
                {
                    "vertex_id": v.vertex_id,
                    "kind": v.kind,
                    "op_id": list(v.op_id) if v.op_id is not None else None,
                }
                for v in self.topological_order
            ],
            "operations": [
                {
                    "job": op.job,
                    "op_index": op.op_index,
                    "vertex_id": op.vertex_id,
                    "machine": op.machine,
                    "processing_time": op.processing_time,
                    "head": op.head,
                    "tail": op.tail,
                    "slack": op.slack,
                    "is_critical": op.is_critical,
                    "maximizing_predecessors": [list(p) for p in op.maximizing_predecessors],
                    "representative_predecessor": (
                        list(op.representative_predecessor)
                        if op.representative_predecessor is not None
                        else None
                    ),
                    "maximizing_successors": [list(s) for s in op.maximizing_successors],
                    "representative_successor": (
                        list(op.representative_successor)
                        if op.representative_successor is not None
                        else None
                    ),
                }
                for op in self.operations
            ],
            "arcs": [
                {
                    "arc_id": arc.arc_id,
                    "from_vertex": arc.from_vertex,
                    "to_vertex": arc.to_vertex,
                    "from_op": list(arc.from_op) if arc.from_op is not None else None,
                    "to_op": list(arc.to_op) if arc.to_op is not None else None,
                    "arc_type": arc.arc_type,
                    "owner_id": arc.owner_id,
                    "is_head_tight": arc.is_head_tight,
                    "is_critical": arc.is_critical,
                }
                for arc in self.arcs
            ],
            "critical_operations": [list(op) for op in self.critical_operations],
            "critical_arc_ids": list(self.critical_arc_ids),
            "critical_machine_arc_ids": list(self.critical_machine_arc_ids),
            "critical_job_arc_ids": list(self.critical_job_arc_ids),
            "representative_critical_path": {
                "operations": [list(op) for op in self.representative_critical_path.operations],
                "arc_ids": list(self.representative_critical_path.arc_ids),
                "total_processing_time": self.representative_critical_path.total_processing_time,
            },
            "critical_blocks": [
                {
                    "block_id": blk.block_id,
                    "machine": blk.machine,
                    "operations": [list(op) for op in blk.operations],
                    "first_operation": list(blk.first_operation),
                    "last_operation": list(blk.last_operation),
                    "length": blk.length,
                    "critical_machine_arc_ids": list(blk.critical_machine_arc_ids),
                }
                for blk in self.critical_blocks
            ],
        }


@dataclass(frozen=True, slots=True)
class CanonicalOperationKeyRecord:
    """Canonical `D01-B` sorting key `(head, completion, job, op_index)`."""

    job: int
    op_index: int
    head: int
    completion: int


@dataclass(frozen=True, slots=True)
class EncodedObrResultRecord:
    """Immutable Python container mirroring Rust `EncodedObrResult`."""

    chromosome: tuple[int, ...]
    canonical_operations: tuple[tuple[int, int], ...]
    operation_keys: tuple[CanonicalOperationKeyRecord, ...]
    chromosome_length: int
    job_multiplicities: tuple[int, ...]
    technological_order_preserved: bool
    obr_valid: bool


@dataclass(frozen=True, slots=True)
class AdapterAuditReportRecord:
    """Immutable Python container mirroring Rust `AdapterAuditReport`."""

    target_graph_hash: str
    generated_obr_hash: str
    decoded_schedule_hash: str
    target_graph_makespan: int
    decoded_schedule_makespan: int
    signed_makespan_diff: int
    adapter_left_shift_count: int
    adapter_right_shift_count: int
    adapter_unchanged_start_count: int
    machine_order_inversion_count: int
    mismatched_machine_arc_count: int
    exact_machine_order_preserved: bool
    exact_start_times_preserved: bool
    exact_completion_times_preserved: bool
    obr_valid: bool
    technological_order_preserved: bool
    status: str
    total_operation_count: int

    @property
    def is_exact(self) -> bool:
        """Return `True` iff the round-trip status is strictly `'exact'`."""
        return self.status == "exact"

    def to_dict(self) -> dict[str, object]:
        """Serialize the adapter audit telemetry to JSON-compatible primitives."""
        return {
            "target_graph_hash": self.target_graph_hash,
            "generated_obr_hash": self.generated_obr_hash,
            "decoded_schedule_hash": self.decoded_schedule_hash,
            "target_graph_makespan": self.target_graph_makespan,
            "decoded_schedule_makespan": self.decoded_schedule_makespan,
            "signed_makespan_diff": self.signed_makespan_diff,
            "adapter_left_shift_count": self.adapter_left_shift_count,
            "adapter_right_shift_count": self.adapter_right_shift_count,
            "adapter_unchanged_start_count": self.adapter_unchanged_start_count,
            "machine_order_inversion_count": self.machine_order_inversion_count,
            "mismatched_machine_arc_count": self.mismatched_machine_arc_count,
            "exact_machine_order_preserved": self.exact_machine_order_preserved,
            "exact_start_times_preserved": self.exact_start_times_preserved,
            "exact_completion_times_preserved": self.exact_completion_times_preserved,
            "obr_valid": self.obr_valid,
            "technological_order_preserved": self.technological_order_preserved,
            "status": self.status,
            "total_operation_count": self.total_operation_count,
        }


@dataclass(frozen=True, slots=True)
class AdapterRoundTripResult:
    """Complete Python bundle for an OBR <-> Machine-Sequence adapter round-trip check."""

    encoded: EncodedObrResultRecord
    decoded_schedule: DecodedScheduleResult
    audit: AdapterAuditReportRecord


def _normalize_jobs(jobs: InstanceSpec) -> list[list[tuple[int, int]]]:
    return [[(int(machine), int(duration)) for machine, duration in job_row] for job_row in jobs]


def _payload_to_result(
    payload: tuple[int, list[RawScheduledOperationTuple], list[list[tuple[int, int]]]],
) -> DecodedScheduleResult:
    makespan, raw_ops, raw_machine_seqs = payload
    operations = tuple(
        ScheduledOperationRecord(
            job=job,
            op_index=op_index,
            machine=machine,
            start_time=start_time,
            completion_time=completion_time,
            processing_time=processing_time,
        )
        for job, op_index, machine, start_time, completion_time, processing_time in raw_ops
    )
    machine_sequences = tuple(
        tuple((job, op_index) for job, op_index in machine_seq) for machine_seq in raw_machine_seqs
    )
    return DecodedScheduleResult(
        makespan=int(makespan),
        operations=operations,
        machine_sequences=machine_sequences,
    )


def get_bridge_info() -> str:
    """Return the fixed diagnostic identification string from the Rust extension."""
    return str(_native.bridge_info())


def get_core_version() -> str:
    """Return the compiled `jssp-core` version string from the Rust extension."""
    return str(_native.core_version())


def get_native_bridge_info() -> dict[str, str]:
    """Return structured diagnostic metadata describing the native Rust bridge."""
    return {
        "bridge_info": get_bridge_info(),
        "core_version": get_core_version(),
        "module_version": str(_native.__version__),
    }


def validate_instance(jobs: InstanceSpec) -> tuple[int, int, int, int]:
    """Validate a rectangular JSSP instance in Rust.

    Returns `(num_jobs, num_machines, num_operations, total_processing_time)`.
    """
    num_jobs, num_machines, num_operations, total_processing_time = _native.validate_instance(
        _normalize_jobs(jobs)
    )
    return (
        int(num_jobs),
        int(num_machines),
        int(num_operations),
        int(total_processing_time),
    )


def validate_obr(jobs: InstanceSpec, chromosome: Sequence[int]) -> bool:
    """Validate an OBR chromosome against a JSSP instance in Rust."""
    return bool(_native.validate_obr(_normalize_jobs(jobs), [int(gene) for gene in chromosome]))


def resolve_obr_operations(jobs: InstanceSpec, chromosome: Sequence[int]) -> list[tuple[int, int]]:
    """Validate an OBR chromosome and resolve each position into `(job, op_index)`."""
    resolved = _native.resolve_obr_operations(
        _normalize_jobs(jobs), [int(gene) for gene in chromosome]
    )
    return [(int(job), int(op_index)) for job, op_index in resolved]


def decode_obr(jobs: InstanceSpec, chromosome: Sequence[int]) -> DecodedScheduleResult:
    """Decode an OBR chromosome using canonical Serial SGS earliest idle-gap insertion."""
    payload = _native.decode_obr(_normalize_jobs(jobs), [int(gene) for gene in chromosome])
    return _payload_to_result(payload)


def decode_obr_append_only(jobs: InstanceSpec, chromosome: Sequence[int]) -> DecodedScheduleResult:
    """Decode an OBR chromosome using semi-active append-only placement for test comparison."""
    payload = _native.decode_obr_append_only(
        _normalize_jobs(jobs), [int(gene) for gene in chromosome]
    )
    return _payload_to_result(payload)


def validate_raw_schedule(
    jobs: InstanceSpec,
    operations: Sequence[RawScheduledOperationTuple],
    makespan: int,
) -> bool:
    """Validate raw schedule tuples against classical JSSP feasibility constraints in Rust."""
    raw_ops = [(int(j), int(k), int(m), int(s), int(c), int(p)) for j, k, m, s, c, p in operations]
    return bool(_native.validate_schedule(_normalize_jobs(jobs), raw_ops, int(makespan)))


def validate_schedule(jobs: InstanceSpec, schedule: DecodedScheduleResult) -> bool:
    """Validate a `DecodedScheduleResult` against all classical JSSP feasibility invariants."""
    return validate_raw_schedule(jobs, schedule.to_raw_operations(), schedule.makespan)


def evaluate_makespan(jobs: InstanceSpec, schedule: DecodedScheduleResult) -> int:
    """Independently recompute makespan from scheduled completion times in Rust."""
    return int(
        _native.evaluate_makespan(
            _normalize_jobs(jobs),
            schedule.to_raw_operations(),
            int(schedule.makespan),
        )
    )


def get_frozen_master_seeds() -> tuple[int, ...]:
    """Return the frozen 20-element confirmatory master-seed schedule from Rust."""
    return tuple(int(seed) for seed in _native.frozen_master_seeds())


def get_rng_metadata() -> RngMetadataRecord:
    """Return authoritative RNG algorithm, crate, schema, and role metadata from Rust."""
    algo, crate_ver, hash_ver, rounds, schema, sched_id, roles = _native.rng_metadata()
    return RngMetadataRecord(
        algorithm=str(algo),
        rng_crate_version=str(crate_ver),
        hash_crate_version=str(hash_ver),
        chacha_rounds=int(rounds),
        seed_derivation_schema=str(schema),
        master_seed_schedule_id=str(sched_id),
        stream_roles=tuple(str(r) for r in roles),
    )


def derive_rng_streams(
    replicate_index: int,
    instance_id: str,
    pairing_mode: str,
    pairing_group: str,
    configuration_id: str,
    preview_count: int = 4,
) -> tuple[DerivedStreamPreview, ...]:
    """Derive all nine D07-A named ChaCha20 RNG streams for a frozen replicate in Rust."""
    rows = _native.derive_rng_streams(
        int(replicate_index),
        str(instance_id),
        str(pairing_mode),
        str(pairing_group),
        str(configuration_id),
        int(preview_count),
    )
    return tuple(
        DerivedStreamPreview(
            role=str(role),
            coupling_key=str(coupling_key),
            seed_hex=str(seed_hex),
            preview_draws_u64=tuple(int(w) for w in draws),
        )
        for role, coupling_key, seed_hex, draws in rows
    )


def derive_rng_streams_with_seed(
    master_seed: int,
    replicate_index: int,
    instance_id: str,
    pairing_mode: str,
    pairing_group: str,
    configuration_id: str,
    preview_count: int = 4,
) -> tuple[DerivedStreamPreview, ...]:
    """Derive all nine D07-A named ChaCha20 RNG streams for an explicit 64-bit master seed."""
    rows = _native.derive_rng_streams_with_seed(
        int(master_seed),
        int(replicate_index),
        str(instance_id),
        str(pairing_mode),
        str(pairing_group),
        str(configuration_id),
        int(preview_count),
    )
    return tuple(
        DerivedStreamPreview(
            role=str(role),
            coupling_key=str(coupling_key),
            seed_hex=str(seed_hex),
            preview_draws_u64=tuple(int(w) for w in draws),
        )
        for role, coupling_key, seed_hex, draws in rows
    )


def rng_sample_range_and_shuffle(
    seed_hex: str,
    upper: int,
    count: int,
    shuffle_len: int,
) -> tuple[list[int], list[int]]:
    """Run Rust ChaCha20 unbiased integer sampling and Fisher-Yates shuffle from `seed_hex`."""
    samples, permutation = _native.rng_sample_range_and_shuffle(
        str(seed_hex),
        int(upper),
        int(count),
        int(shuffle_len),
    )
    return ([int(x) for x in samples], [int(i) for i in permutation])


def _pair_or_none(value: tuple[int, int] | None) -> tuple[int, int] | None:
    if value is None:
        return None
    return (int(value[0]), int(value[1]))


def _graph_payload_to_result(
    payload: tuple[
        int,
        int,
        tuple[int, int, int, int, int, int, int, int, int],
        list[tuple[int, str, tuple[int, int] | None]],
        list[
            tuple[
                tuple[int, int],
                int,
                int,
                int,
                tuple[int, int, int, bool],
                list[tuple[int, int]],
                tuple[int, int] | None,
                list[tuple[int, int]],
                tuple[int, int] | None,
            ]
        ],
        list[
            tuple[
                int,
                int,
                int,
                tuple[int, int] | None,
                tuple[int, int] | None,
                str,
                int | None,
                bool,
                bool,
            ]
        ],
        tuple[list[tuple[int, int]], list[int], list[int], list[int]],
        tuple[list[tuple[int, int]], list[int], int],
        list[
            tuple[
                int,
                int,
                list[tuple[int, int]],
                tuple[int, int],
                tuple[int, int],
                int,
                list[int],
            ]
        ],
    ],
) -> DisjunctiveGraphAnalysisResult:
    (
        graph_makespan,
        schedule_makespan,
        meta_raw,
        topo_raw,
        ops_raw,
        arcs_raw,
        subgraph_raw,
        rep_path_raw,
        blocks_raw,
    ) = payload

    metadata = GraphMetadataRecord(
        num_jobs=int(meta_raw[0]),
        num_machines=int(meta_raw[1]),
        num_operations=int(meta_raw[2]),
        num_vertices=int(meta_raw[3]),
        num_arcs=int(meta_raw[4]),
        num_job_precedence_arcs=int(meta_raw[5]),
        num_machine_order_arcs=int(meta_raw[6]),
        num_source_arcs=int(meta_raw[7]),
        num_sink_arcs=int(meta_raw[8]),
    )
    topological_order = tuple(
        GraphVertexRecord(vertex_id=int(v_id), kind=str(kind), op_id=_pair_or_none(op))
        for v_id, kind, op in topo_raw
    )
    operations = tuple(
        OperationCriticalityRecord(
            job=int(op_pair[0]),
            op_index=int(op_pair[1]),
            vertex_id=int(v_id),
            machine=int(machine),
            processing_time=int(proc_time),
            head=int(timing[0]),
            tail=int(timing[1]),
            slack=int(timing[2]),
            is_critical=bool(timing[3]),
            maximizing_predecessors=tuple((int(j), int(k)) for j, k in max_preds),
            representative_predecessor=_pair_or_none(rep_pred),
            maximizing_successors=tuple((int(j), int(k)) for j, k in max_succs),
            representative_successor=_pair_or_none(rep_succ),
        )
        for (
            op_pair,
            v_id,
            machine,
            proc_time,
            timing,
            max_preds,
            rep_pred,
            max_succs,
            rep_succ,
        ) in ops_raw
    )
    arcs = tuple(
        GraphArcRecord(
            arc_id=int(arc_id),
            from_vertex=int(from_v),
            to_vertex=int(to_v),
            from_op=_pair_or_none(from_op),
            to_op=_pair_or_none(to_op),
            arc_type=str(arc_type),
            owner_id=int(owner_id) if owner_id is not None else None,
            is_head_tight=bool(is_tight),
            is_critical=bool(is_crit),
        )
        for (
            arc_id,
            from_v,
            to_v,
            from_op,
            to_op,
            arc_type,
            owner_id,
            is_tight,
            is_crit,
        ) in arcs_raw
    )
    crit_ops_raw, crit_arcs_raw, crit_machine_arcs_raw, crit_job_arcs_raw = subgraph_raw
    rep_ops_raw, rep_arcs_raw, rep_total_time = rep_path_raw
    representative_critical_path = RepresentativeCriticalPathRecord(
        operations=tuple((int(j), int(k)) for j, k in rep_ops_raw),
        arc_ids=tuple(int(a) for a in rep_arcs_raw),
        total_processing_time=int(rep_total_time),
    )
    critical_blocks = tuple(
        CriticalBlockRecord(
            block_id=int(b_id),
            machine=int(machine),
            operations=tuple((int(j), int(k)) for j, k in ops),
            first_operation=(int(first_op[0]), int(first_op[1])),
            last_operation=(int(last_op[0]), int(last_op[1])),
            length=int(length),
            critical_machine_arc_ids=tuple(int(a) for a in arc_ids),
        )
        for b_id, machine, ops, first_op, last_op, length, arc_ids in blocks_raw
    )

    return DisjunctiveGraphAnalysisResult(
        graph_makespan=int(graph_makespan),
        schedule_makespan=int(schedule_makespan),
        metadata=metadata,
        topological_order=topological_order,
        operations=operations,
        arcs=arcs,
        critical_operations=tuple((int(j), int(k)) for j, k in crit_ops_raw),
        critical_arc_ids=tuple(int(a) for a in crit_arcs_raw),
        critical_machine_arc_ids=tuple(int(a) for a in crit_machine_arcs_raw),
        critical_job_arc_ids=tuple(int(a) for a in crit_job_arcs_raw),
        representative_critical_path=representative_critical_path,
        critical_blocks=critical_blocks,
    )


def analyze_raw_schedule_graph(
    jobs: InstanceSpec,
    operations: Sequence[RawScheduledOperationTuple],
    makespan: int,
) -> DisjunctiveGraphAnalysisResult:
    """Build and analyze the compact oriented disjunctive graph from raw schedule tuples."""
    raw_ops = [(int(j), int(k), int(m), int(s), int(c), int(p)) for j, k, m, s, c, p in operations]
    payload = _native.analyze_schedule_graph(_normalize_jobs(jobs), raw_ops, int(makespan))
    return _graph_payload_to_result(payload)


def analyze_schedule_graph(
    jobs: InstanceSpec,
    schedule: DecodedScheduleResult,
) -> DisjunctiveGraphAnalysisResult:
    """Build the compact oriented disjunctive graph and run critical-path analysis in Rust."""
    return analyze_raw_schedule_graph(jobs, schedule.to_raw_operations(), schedule.makespan)


def build_disjunctive_graph(
    jobs: InstanceSpec,
    schedule: DecodedScheduleResult,
) -> DisjunctiveGraphAnalysisResult:
    """Construct and analyze a validated disjunctive graph from a schedule."""
    return analyze_schedule_graph(jobs, schedule)


def analyze_obr_graph(
    jobs: InstanceSpec,
    chromosome: Sequence[int],
) -> DisjunctiveGraphAnalysisResult:
    """Decode an OBR chromosome and analyze its disjunctive graph & critical blocks in Rust."""
    payload = _native.analyze_obr_graph(_normalize_jobs(jobs), [int(gene) for gene in chromosome])
    return _graph_payload_to_result(payload)


def compute_head_tail_times(
    jobs: InstanceSpec,
    schedule: DecodedScheduleResult,
) -> tuple[OperationCriticalityRecord, ...]:
    """Compute head, tail, slack, and maximizing predecessor/successor ties in Rust."""
    return analyze_schedule_graph(jobs, schedule).operations


def get_critical_operations(
    jobs: InstanceSpec,
    schedule: DecodedScheduleResult,
) -> tuple[tuple[int, int], ...]:
    """Return all critical `(job, op_index)` pairs from Rust critical-path analysis."""
    return analyze_schedule_graph(jobs, schedule).critical_operations


def get_critical_arcs(
    jobs: InstanceSpec,
    schedule: DecodedScheduleResult,
) -> tuple[GraphArcRecord, ...]:
    """Return all critical arcs (Boundary, JobPrecedence, MachineOrder) from Rust."""
    return analyze_schedule_graph(jobs, schedule).critical_arcs


def get_critical_machine_arcs(
    jobs: InstanceSpec,
    schedule: DecodedScheduleResult,
) -> tuple[GraphArcRecord, ...]:
    """Return all critical machine-order arcs from Rust."""
    return analyze_schedule_graph(jobs, schedule).critical_machine_arcs


def get_representative_critical_path(
    jobs: InstanceSpec,
    schedule: DecodedScheduleResult,
) -> RepresentativeCriticalPathRecord:
    """Return the deterministic representative Source-to-Sink critical path from Rust."""
    return analyze_schedule_graph(jobs, schedule).representative_critical_path


def get_critical_blocks(
    jobs: InstanceSpec,
    schedule: DecodedScheduleResult,
) -> tuple[CriticalBlockRecord, ...]:
    """Return all maximal extracted machine critical blocks (`length >= 2`) from Rust."""
    return analyze_schedule_graph(jobs, schedule).critical_blocks


def get_graph_makespan(
    jobs: InstanceSpec,
    schedule: DecodedScheduleResult,
) -> int:
    """Return the longest Source-to-Sink path length verified against schedule makespan."""
    return analyze_schedule_graph(jobs, schedule).graph_makespan


def validate_machine_sequences_dag(
    jobs: InstanceSpec,
    machine_sequences: Sequence[Sequence[tuple[int, int]]],
) -> tuple[GraphVertexRecord, ...]:
    """Verify machine-sequence acyclicity in Rust and return topological vertex ordering."""
    seqs = [[(int(j), int(k)) for j, k in seq] for seq in machine_sequences]
    topo_raw = _native.validate_machine_sequences_dag(_normalize_jobs(jobs), seqs)
    return tuple(
        GraphVertexRecord(vertex_id=int(v_id), kind=str(kind), op_id=_pair_or_none(op))
        for v_id, kind, op in topo_raw
    )


def _encoded_payload_to_record(
    payload: tuple[
        list[int],
        list[tuple[int, int]],
        list[tuple[int, int, int, int]],
        int,
        list[int],
        bool,
        bool,
    ],
) -> EncodedObrResultRecord:
    genes, canon_ops, op_keys, chrom_len, multiplicities, tech_ok, obr_valid = payload
    return EncodedObrResultRecord(
        chromosome=tuple(int(g) for g in genes),
        canonical_operations=tuple((int(j), int(k)) for j, k in canon_ops),
        operation_keys=tuple(
            CanonicalOperationKeyRecord(job=int(j), op_index=int(k), head=int(h), completion=int(c))
            for j, k, h, c in op_keys
        ),
        chromosome_length=int(chrom_len),
        job_multiplicities=tuple(int(m) for m in multiplicities),
        technological_order_preserved=bool(tech_ok),
        obr_valid=bool(obr_valid),
    )


def _audit_payload_to_record(
    payload: tuple[
        tuple[str, str, str],
        tuple[int, int, int],
        tuple[int, int, int],
        tuple[int, int, bool],
        tuple[bool, bool, bool, bool],
        str,
        int,
    ],
) -> AdapterAuditReportRecord:
    hashes, makespans, shifts, machine_diff, preservation, status, total_ops = payload
    return AdapterAuditReportRecord(
        target_graph_hash=str(hashes[0]),
        generated_obr_hash=str(hashes[1]),
        decoded_schedule_hash=str(hashes[2]),
        target_graph_makespan=int(makespans[0]),
        decoded_schedule_makespan=int(makespans[1]),
        signed_makespan_diff=int(makespans[2]),
        adapter_left_shift_count=int(shifts[0]),
        adapter_right_shift_count=int(shifts[1]),
        adapter_unchanged_start_count=int(shifts[2]),
        machine_order_inversion_count=int(machine_diff[0]),
        mismatched_machine_arc_count=int(machine_diff[1]),
        exact_machine_order_preserved=bool(machine_diff[2]),
        exact_start_times_preserved=bool(preservation[0]),
        exact_completion_times_preserved=bool(preservation[1]),
        obr_valid=bool(preservation[2]),
        technological_order_preserved=bool(preservation[3]),
        status=str(status),
        total_operation_count=int(total_ops),
    )


def build_machine_sequence_graph(
    jobs: InstanceSpec,
    machine_sequences: Sequence[Sequence[tuple[int, int]]],
) -> DisjunctiveGraphAnalysisResult:
    """Validate machine sequences, build an acyclic DAG, and compute critical-path analysis."""
    seqs = [[(int(j), int(k)) for j, k in seq] for seq in machine_sequences]
    payload = _native.build_machine_sequence_graph(_normalize_jobs(jobs), seqs)
    return _graph_payload_to_result(payload)


def encode_machine_sequences_to_obr(
    jobs: InstanceSpec,
    machine_sequences: Sequence[Sequence[tuple[int, int]]],
) -> EncodedObrResultRecord:
    """Encode a validated machine-sequence graph into a canonical `D01-B` OBR chromosome."""
    seqs = [[(int(j), int(k)) for j, k in seq] for seq in machine_sequences]
    payload = _native.encode_machine_sequences_to_obr(_normalize_jobs(jobs), seqs)
    return _encoded_payload_to_record(payload)


def round_trip_machine_sequences(
    jobs: InstanceSpec,
    machine_sequences: Sequence[Sequence[tuple[int, int]]],
) -> AdapterRoundTripResult:
    """Validate machine sequences, encode to OBR, decode with Serial SGS, and return audit."""
    seqs = [[(int(j), int(k)) for j, k in seq] for seq in machine_sequences]
    encoded_raw, schedule_raw, audit_raw = _native.round_trip_machine_sequences(
        _normalize_jobs(jobs), seqs
    )
    return AdapterRoundTripResult(
        encoded=_encoded_payload_to_record(encoded_raw),
        decoded_schedule=_payload_to_result(schedule_raw),
        audit=_audit_payload_to_record(audit_raw),
    )


def round_trip_obr_schedule(
    jobs: InstanceSpec,
    chromosome: Sequence[int],
) -> AdapterRoundTripResult:
    """Decode an OBR chromosome via M2 Serial SGS and audit its canonical D01-B round trip."""
    encoded_raw, schedule_raw, audit_raw = _native.round_trip_obr_schedule(
        _normalize_jobs(jobs), [int(gene) for gene in chromosome]
    )
    return AdapterRoundTripResult(
        encoded=_encoded_payload_to_record(encoded_raw),
        decoded_schedule=_payload_to_result(schedule_raw),
        audit=_audit_payload_to_record(audit_raw),
    )


@dataclass(frozen=True, slots=True)
class PhaseAInitializerSpec:
    """Metadata describing one Phase A initial population configuration (`A01`–`A24`)."""

    code: str
    name: str
    alpha: float | None
    is_deterministic: bool
    uses_sobol: bool
    has_dyadic_sobol_net_balance: bool


@dataclass(frozen=True, slots=True)
class PopulationDiagnosticsRecord:
    """Immutable Layer A / final-population diagnostic statistics returned from Rust."""

    population_size: int
    feasible_count: int
    unique_chromosomes: int
    unique_schedules: int
    best_makespan: int
    worst_makespan: int
    mean_makespan: float
    median_makespan: float
    std_makespan: float
    obr_duplicate_rate: float
    schedule_duplicate_rate: float
    obr_hamming_diversity: float
    schedule_diversity: float
    qmc_1d_ks_discrepancy: float | None
    generation_elapsed_ns: int
    decode_elapsed_ns: int


@dataclass(frozen=True, slots=True)
class GenerationSnapshotRecord:
    """Single generation telemetry record from a Classical GA (`C01`) trajectory."""

    generation: int
    evals_used: int
    best_makespan: int
    mean_makespan: float
    obr_hamming_diversity: float
    unique_chromosomes: int


@dataclass(frozen=True, slots=True)
class InitializerEvaluationResult:
    """Output from standalone Layer A initial population evaluation."""

    diagnostics: PopulationDiagnosticsRecord
    chromosomes: tuple[tuple[int, ...], ...]
    makespans: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PhaseARunResultRecord:
    """Complete Layer A + Layer B telemetry bundle from one `(initializer, C01)` GA run."""

    initializer_id: str
    pop_size: int
    evaluation_budget: int
    evals_used: int
    generations_executed: int
    best_makespan: int
    best_found_eval: int
    best_found_generation: int
    best_schedule_hash: str
    anytime_cmax_auc: float
    crossover_count: int
    mutation_count: int
    total_elapsed_ns: int
    initial: PopulationDiagnosticsRecord
    final_pop: PopulationDiagnosticsRecord
    best_chromosome: tuple[int, ...]
    trajectory: tuple[GenerationSnapshotRecord, ...]


def _diag_payload_to_record(
    payload: tuple[
        tuple[int, int, int, int],
        tuple[int, int, float, float, float],
        tuple[float, float, float, float, float | None],
        tuple[int, int],
    ],
) -> PopulationDiagnosticsRecord:
    counts, cmax, div, timing = payload
    return PopulationDiagnosticsRecord(
        population_size=int(counts[0]),
        feasible_count=int(counts[1]),
        unique_chromosomes=int(counts[2]),
        unique_schedules=int(counts[3]),
        best_makespan=int(cmax[0]),
        worst_makespan=int(cmax[1]),
        mean_makespan=float(cmax[2]),
        median_makespan=float(cmax[3]),
        std_makespan=float(cmax[4]),
        obr_duplicate_rate=float(div[0]),
        schedule_duplicate_rate=float(div[1]),
        obr_hamming_diversity=float(div[2]),
        schedule_diversity=float(div[3]),
        qmc_1d_ks_discrepancy=(float(div[4]) if div[4] is not None else None),
        generation_elapsed_ns=int(timing[0]),
        decode_elapsed_ns=int(timing[1]),
    )


def _phase_a_run_payload_to_record(
    payload: tuple[
        tuple[str, int, int, int, int, int, int, int, str, float],
        tuple[int, int, int],
        tuple,
        tuple,
        list[int],
        list[tuple[int, int, int, float, float, int]],
    ],
) -> PhaseARunResultRecord:
    summary, op_counts, init_raw, final_raw, best_chr, traj_raw = payload
    return PhaseARunResultRecord(
        initializer_id=str(summary[0]),
        pop_size=int(summary[1]),
        evaluation_budget=int(summary[2]),
        evals_used=int(summary[3]),
        generations_executed=int(summary[4]),
        best_makespan=int(summary[5]),
        best_found_eval=int(summary[6]),
        best_found_generation=int(summary[7]),
        best_schedule_hash=str(summary[8]),
        anytime_cmax_auc=float(summary[9]),
        crossover_count=int(op_counts[0]),
        mutation_count=int(op_counts[1]),
        total_elapsed_ns=int(op_counts[2]),
        initial=_diag_payload_to_record(init_raw),
        final_pop=_diag_payload_to_record(final_raw),
        best_chromosome=tuple(int(g) for g in best_chr),
        trajectory=tuple(
            GenerationSnapshotRecord(
                generation=int(g),
                evals_used=int(ev),
                best_makespan=int(bm),
                mean_makespan=float(mm),
                obr_hamming_diversity=float(hd),
                unique_chromosomes=int(uc),
            )
            for g, ev, bm, mm, hd, uc in traj_raw
        ),
    )


def list_phase_a_initializers() -> tuple[PhaseAInitializerSpec, ...]:
    """Return authoritative Rust metadata across all 24 Phase A initializers (`A01`–`A24`)."""
    return tuple(
        PhaseAInitializerSpec(
            code=str(code),
            name=str(name),
            alpha=(float(alpha) if alpha is not None else None),
            is_deterministic=bool(det),
            uses_sobol=bool(sobol),
            has_dyadic_sobol_net_balance=bool(dyadic),
        )
        for code, name, alpha, det, sobol, dyadic in _native.list_phase_a_initializers()
    )


def default_phase_a_ga_budget(num_jobs: int, num_machines: int) -> int:
    """Return the size-scaled Phase A Classical GA evaluation budget from Rust."""
    return int(_native.default_phase_a_ga_budget(int(num_jobs), int(num_machines)))


def evaluate_phase_a_initializer(
    jobs: InstanceSpec,
    initializer_id: str,
    instance_id: str,
    *,
    pop_size: int = 64,
    replicate_index: int = 0,
    master_seed_override: int | None = None,
    pairing_mode: str = "paired_init",
    pairing_group: str = "phase_a_shared_init",
    configuration_id: str | None = None,
) -> InitializerEvaluationResult:
    """Generate and decode one standalone Layer A initial population in Rust."""
    diag_raw, chrs_raw, cmax_raw = _native.evaluate_phase_a_initializer(
        _normalize_jobs(jobs),
        initializer_id,
        instance_id,
        int(pop_size),
        int(replicate_index),
        master_seed_override,
        pairing_mode,
        pairing_group,
        configuration_id,
    )
    return InitializerEvaluationResult(
        diagnostics=_diag_payload_to_record(diag_raw),
        chromosomes=tuple(tuple(int(g) for g in chr_row) for chr_row in chrs_raw),
        makespans=tuple(int(m) for m in cmax_raw),
    )


def run_phase_a_single(
    jobs: InstanceSpec,
    initializer_id: str,
    instance_id: str,
    *,
    pop_size: int = 64,
    tournament_size: int = 2,
    crossover_rate: float = 0.85,
    mutation_rate: float = 0.15,
    elitism_count: int = 2,
    evaluation_budget: int | None = None,
    replicate_index: int = 0,
    master_seed_override: int | None = None,
    pairing_mode: str = "paired_init",
    pairing_group: str = "phase_a_shared_init",
    configuration_id: str | None = None,
    trajectory_mode: str = "full",
) -> PhaseARunResultRecord:
    """Run one Phase A `(initializer -> Classical GA C01)` experiment cleanly inside Rust."""
    payload = _native.run_phase_a_single(
        _normalize_jobs(jobs),
        initializer_id,
        instance_id,
        int(pop_size),
        int(tournament_size),
        float(crossover_rate),
        float(mutation_rate),
        int(elitism_count),
        (int(evaluation_budget) if evaluation_budget is not None else None),
        int(replicate_index),
        master_seed_override,
        pairing_mode,
        pairing_group,
        configuration_id,
        trajectory_mode,
    )
    return _phase_a_run_payload_to_record(payload)


def run_phase_a_batch(
    jobs: InstanceSpec,
    instance_id: str,
    tasks: Sequence[tuple[str, int, int | None, str, str, str]],
    *,
    pop_size: int = 64,
    tournament_size: int = 2,
    crossover_rate: float = 0.85,
    mutation_rate: float = 0.15,
    elitism_count: int = 2,
    evaluation_budget: int | None = None,
    num_threads: int = 0,
    trajectory_mode: str = "sampled",
) -> tuple[PhaseARunResultRecord, ...]:
    """Run a multi-threaded Rust batch of Phase A `(initializer, replicate)` tasks."""
    normalized_tasks = [
        (str(init_id), int(rep), seed_ov, str(pmode), str(pgroup), str(cfg_id))
        for init_id, rep, seed_ov, pmode, pgroup, cfg_id in tasks
    ]
    raw_list = _native.run_phase_a_batch(
        _normalize_jobs(jobs),
        instance_id,
        normalized_tasks,
        int(pop_size),
        int(tournament_size),
        float(crossover_rate),
        float(mutation_rate),
        int(elitism_count),
        (int(evaluation_budget) if evaluation_budget is not None else None),
        int(num_threads),
        trajectory_mode,
    )
    return tuple(_phase_a_run_payload_to_record(item) for item in raw_list)


@dataclass(frozen=True, slots=True)
class PhaseBConfigSpec:
    """Metadata describing one active Phase B local-search or metaheuristic configuration."""

    code: str
    strategy_name: str
    neighborhood_code: str


@dataclass(frozen=True, slots=True)
class SearchCountersRecord:
    """Cumulative Phase B move, feasibility, cycle-rejection, and perturbation counters."""

    iterations: int
    generated_moves: int
    duplicate_pruned_moves: int
    infeasible_precond_moves: int
    clipped_moves: int
    evaluated_moves: int
    feasible_moves: int
    rejected_cyclic_moves: int
    improving_moves: int
    accepted_moves: int
    adapter_left_shift_count: int
    perturbations_applied: int
    perturbations_accepted: int
    restarts_triggered: int


@dataclass(frozen=True, slots=True)
class PhaseBRunResultRecord:
    """Complete telemetry outcome returned from one Phase B run in Rust."""

    config_id: str
    neighborhood_id: str
    evaluation_budget: int
    evals_used: int
    initial_makespan: int
    final_makespan: int
    best_makespan: int
    best_found_eval: int
    feasible_final: bool
    reached_local_optimum: bool
    terminated_reason: str
    anytime_cmax_auc: float
    total_elapsed_ns: int
    counters: SearchCountersRecord
    best_milestones: tuple[tuple[int, int], ...]
    best_machine_sequences: tuple[tuple[tuple[int, int], ...], ...]


def _phase_b_run_payload_to_record(
    payload: tuple[
        tuple[str, str, int, int, int, int, int, int],
        tuple[bool, bool, str, float, int],
        tuple[int, int, int, int, int, int, int],
        tuple[int, int, int, int, int, int, int],
        list[tuple[int, int]],
        list[list[tuple[int, int]]],
    ],
) -> PhaseBRunResultRecord:
    summary_core, status_meta, c1, c2, milestones, seqs = payload
    counters = SearchCountersRecord(
        iterations=int(c1[0]),
        generated_moves=int(c1[1]),
        duplicate_pruned_moves=int(c1[2]),
        infeasible_precond_moves=int(c1[3]),
        clipped_moves=int(c1[4]),
        evaluated_moves=int(c1[5]),
        feasible_moves=int(c1[6]),
        rejected_cyclic_moves=int(c2[0]),
        improving_moves=int(c2[1]),
        accepted_moves=int(c2[2]),
        adapter_left_shift_count=int(c2[3]),
        perturbations_applied=int(c2[4]),
        perturbations_accepted=int(c2[5]),
        restarts_triggered=int(c2[6]),
    )
    return PhaseBRunResultRecord(
        config_id=str(summary_core[0]),
        neighborhood_id=str(summary_core[1]),
        evaluation_budget=int(summary_core[2]),
        evals_used=int(summary_core[3]),
        initial_makespan=int(summary_core[4]),
        final_makespan=int(summary_core[5]),
        best_makespan=int(summary_core[6]),
        best_found_eval=int(summary_core[7]),
        feasible_final=bool(status_meta[0]),
        reached_local_optimum=bool(status_meta[1]),
        terminated_reason=str(status_meta[2]),
        anytime_cmax_auc=float(status_meta[3]),
        total_elapsed_ns=int(status_meta[4]),
        counters=counters,
        best_milestones=tuple((int(ev), int(cm)) for ev, cm in milestones),
        best_machine_sequences=tuple(tuple((int(j), int(op)) for j, op in m_seq) for m_seq in seqs),
    )


def list_phase_b_configs() -> tuple[PhaseBConfigSpec, ...]:
    """Return authoritative Rust metadata across all 28 active Phase B configurations."""
    return tuple(
        PhaseBConfigSpec(code=str(c), strategy_name=str(s), neighborhood_code=str(n))
        for c, s, n in _native.list_phase_b_configs()
    )


def default_phase_b_budget(num_jobs: int, num_machines: int, multiplier: int = 100) -> int:
    """Return Phase B candidate-evaluation budget `multiplier * J * M` from Rust."""
    return int(_native.default_phase_b_budget(int(num_jobs), int(num_machines), int(multiplier)))


def generate_matched_phase_b_start(
    jobs: InstanceSpec,
    *,
    initializer_id: str = "A07",
    instance_id: str = "instance",
    pop_size: int = 64,
    replicate_index: int = 0,
    master_seed_override: int | None = None,
    pairing_mode: str = "paired_init",
    pairing_group: str = "phase_b_matched_start",
    configuration_id: str | None = None,
) -> tuple[tuple[int, ...], DecodedScheduleResult]:
    """Generate and decode the matched Phase B starting schedule in Rust."""
    genes_raw, sched_raw = _native.generate_matched_phase_b_start(
        _normalize_jobs(jobs),
        initializer_id,
        instance_id,
        int(pop_size),
        int(replicate_index),
        master_seed_override,
        pairing_mode,
        pairing_group,
        configuration_id,
    )
    return tuple(int(g) for g in genes_raw), _payload_to_result(sched_raw)


def run_phase_b_single(
    jobs: InstanceSpec,
    config_id: str,
    instance_id: str,
    *,
    initializer_id: str = "A07",
    init_pop_size: int = 64,
    evaluation_budget: int | None = None,
    budget_multiplier: int = 100,
    replicate_index: int = 0,
    master_seed_override: int | None = None,
    pairing_mode: str = "paired_init",
    pairing_group: str = "phase_b_matched_start",
    configuration_id: str | None = None,
    start_chromosome: Sequence[int] | None = None,
) -> PhaseBRunResultRecord:
    """Execute one Phase B local-search or metaheuristic run in Rust."""
    payload = _native.run_phase_b_single(
        _normalize_jobs(jobs),
        config_id,
        instance_id,
        initializer_id,
        int(init_pop_size),
        (int(evaluation_budget) if evaluation_budget is not None else None),
        int(budget_multiplier),
        int(replicate_index),
        master_seed_override,
        pairing_mode,
        pairing_group,
        configuration_id,
        ([int(g) for g in start_chromosome] if start_chromosome is not None else None),
    )
    return _phase_b_run_payload_to_record(payload)


def run_phase_b_batch(
    jobs: InstanceSpec,
    instance_id: str,
    tasks: Sequence[tuple[str, int, int | None, str, str, str]],
    *,
    initializer_id: str = "A07",
    init_pop_size: int = 64,
    evaluation_budget: int | None = None,
    budget_multiplier: int = 100,
    num_threads: int = 0,
) -> tuple[PhaseBRunResultRecord, ...]:
    """Run a multi-threaded Rust batch of Phase B `(config_id, replicate)` tasks."""
    normalized_tasks = [
        (str(cfg_id), int(rep), seed_ov, str(pmode), str(pgroup), str(cfg_label))
        for cfg_id, rep, seed_ov, pmode, pgroup, cfg_label in tasks
    ]
    raw_list = _native.run_phase_b_batch(
        _normalize_jobs(jobs),
        instance_id,
        normalized_tasks,
        initializer_id,
        int(init_pop_size),
        (int(evaluation_budget) if evaluation_budget is not None else None),
        int(budget_multiplier),
        int(num_threads),
    )
    return tuple(_phase_b_run_payload_to_record(item) for item in raw_list)


@dataclass(frozen=True, slots=True)
class PhaseCConfigSpec:
    """Metadata describing one Phase C Memetic or Dual-View GA configuration (`C01`–`C11`)."""

    code: str
    name: str
    is_memetic: bool
    is_dual_view_paired: bool


@dataclass(frozen=True, slots=True)
class PhaseCRunResultRecord:
    """Complete telemetry bundle returned from one Phase C (`C01`–`C11`) run in Rust."""

    config_id: str
    initializer_id: str
    ls_neighborhood: str | None
    ls_strategy: str | None
    genotype_capacity: int
    evaluation_budget: int
    evals_used: int
    init_evals: int
    search_evals: int
    decode_evals: int
    ls_evals: int
    generations_executed: int
    best_makespan: int
    best_found_eval: int
    best_found_generation: int
    feasible_final: bool
    anytime_cmax_auc: float
    total_elapsed_ns: int
    final_obr_diversity: float
    final_lam_diversity: float | None
    final_bald_diversity: float | None
    final_gate_p: float | None
    lam_best_makespan: int | None
    bald_best_makespan: int | None
    crossover_count: int
    mutation_count: int
    writeback_count: int
    migrations_applied: int
    gate_p_trajectory: tuple[tuple[int, float], ...]
    best_milestones: tuple[tuple[int, int], ...]
    best_chromosome: tuple[int, ...]
    best_machine_sequences: tuple[tuple[tuple[int, int], ...], ...]


@dataclass(frozen=True, slots=True)
class FinalIntegratedRunRecord:
    """Telemetry returned by the combined `P5 (A* -> C* -> B*)` Rust pipeline runner."""

    pipeline_id: str
    total_budget: int
    evals_used: int
    stage_c_best_makespan: int
    final_best_makespan: int
    stage_c_best_eval: int
    final_best_eval: int
    feasible_final: bool
    anytime_cmax_auc: float
    total_elapsed_ns: int


def _phase_c_run_payload_to_record(
    payload: tuple[
        tuple[str, str, str | None, str | None, int, int, int, int, int],
        tuple[int, int, int, int, int, int, bool, float, int],
        tuple[
            float,
            float | None,
            float | None,
            float | None,
            int | None,
            int | None,
            int,
            int,
            int,
            int,
        ],
        list[tuple[int, float]],
        list[tuple[int, int]],
        list[int],
        list[list[tuple[int, int]]],
    ],
) -> PhaseCRunResultRecord:
    sc, sp, d, gate_traj, milestones, best_chr, seqs = payload
    return PhaseCRunResultRecord(
        config_id=str(sc[0]),
        initializer_id=str(sc[1]),
        ls_neighborhood=(str(sc[2]) if sc[2] is not None else None),
        ls_strategy=(str(sc[3]) if sc[3] is not None else None),
        genotype_capacity=int(sc[4]),
        evaluation_budget=int(sc[5]),
        evals_used=int(sc[6]),
        init_evals=int(sc[7]),
        search_evals=int(sc[8]),
        decode_evals=int(sp[0]),
        ls_evals=int(sp[1]),
        generations_executed=int(sp[2]),
        best_makespan=int(sp[3]),
        best_found_eval=int(sp[4]),
        best_found_generation=int(sp[5]),
        feasible_final=bool(sp[6]),
        anytime_cmax_auc=float(sp[7]),
        total_elapsed_ns=int(sp[8]),
        final_obr_diversity=float(d[0]),
        final_lam_diversity=(float(d[1]) if d[1] is not None else None),
        final_bald_diversity=(float(d[2]) if d[2] is not None else None),
        final_gate_p=(float(d[3]) if d[3] is not None else None),
        lam_best_makespan=(int(d[4]) if d[4] is not None else None),
        bald_best_makespan=(int(d[5]) if d[5] is not None else None),
        crossover_count=int(d[6]),
        mutation_count=int(d[7]),
        writeback_count=int(d[8]),
        migrations_applied=int(d[9]),
        gate_p_trajectory=tuple((int(g), float(p)) for g, p in gate_traj),
        best_milestones=tuple((int(ev), int(cm)) for ev, cm in milestones),
        best_chromosome=tuple(int(x) for x in best_chr),
        best_machine_sequences=tuple(tuple((int(j), int(op)) for j, op in ms) for ms in seqs),
    )


def list_phase_c_configs() -> tuple[PhaseCConfigSpec, ...]:
    """Return authoritative Rust metadata across all 11 Phase C configurations (`C01`–`C11`)."""
    return tuple(
        PhaseCConfigSpec(
            code=str(code),
            name=str(name),
            is_memetic=bool(mem),
            is_dual_view_paired=bool(dv),
        )
        for code, name, mem, dv in _native.list_phase_c_configs()
    )


def default_phase_c_budget(num_jobs: int, num_machines: int) -> int:
    """Return equalized Phase C objective-evaluation budget matching Phase A (`B_GA`)."""
    return int(_native.default_phase_c_budget(int(num_jobs), int(num_machines)))


def run_phase_c_single(
    jobs: InstanceSpec,
    config_id: str,
    instance_id: str,
    *,
    initializer_id: str = "A07",
    ls_neighborhood: str = "N7",
    ls_strategy: str = "steepest_descent",
    max_ls_evals: int = 50,
    pop_size: int = 64,
    tournament_size: int = 2,
    crossover_rate: float = 0.85,
    mutation_rate: float = 0.15,
    elitism_count: int = 2,
    evaluation_budget: int | None = None,
    replicate_index: int = 0,
    master_seed_override: int | None = None,
    pairing_mode: str = "paired_init",
    pairing_group: str = "phase_c_shared_init",
    configuration_id: str | None = None,
) -> PhaseCRunResultRecord:
    """Execute one Phase C (`C01`–`C11`) run in Rust."""
    payload = _native.run_phase_c_single(
        _normalize_jobs(jobs),
        config_id,
        instance_id,
        initializer_id,
        ls_neighborhood,
        ls_strategy,
        int(max_ls_evals),
        int(pop_size),
        int(tournament_size),
        float(crossover_rate),
        float(mutation_rate),
        int(elitism_count),
        (int(evaluation_budget) if evaluation_budget is not None else None),
        int(replicate_index),
        master_seed_override,
        pairing_mode,
        pairing_group,
        configuration_id,
    )
    return _phase_c_run_payload_to_record(payload)


def run_phase_c_batch(
    jobs: InstanceSpec,
    instance_id: str,
    tasks: Sequence[tuple[str, int, int | None, str, str, str]],
    *,
    initializer_id: str = "A07",
    ls_neighborhood: str = "N7",
    ls_strategy: str = "steepest_descent",
    max_ls_evals: int = 50,
    pop_size: int = 64,
    tournament_size: int = 2,
    crossover_rate: float = 0.85,
    mutation_rate: float = 0.15,
    elitism_count: int = 2,
    evaluation_budget: int | None = None,
    num_threads: int = 0,
) -> tuple[PhaseCRunResultRecord, ...]:
    """Run a multi-threaded Rust batch of Phase C `(config_id, replicate)` tasks."""
    normalized_tasks = [
        (str(cfg_id), int(rep), seed_ov, str(pmode), str(pgroup), str(cfg_label))
        for cfg_id, rep, seed_ov, pmode, pgroup, cfg_label in tasks
    ]
    raw_list = _native.run_phase_c_batch(
        _normalize_jobs(jobs),
        instance_id,
        normalized_tasks,
        initializer_id,
        ls_neighborhood,
        ls_strategy,
        int(max_ls_evals),
        int(pop_size),
        int(tournament_size),
        float(crossover_rate),
        float(mutation_rate),
        int(elitism_count),
        (int(evaluation_budget) if evaluation_budget is not None else None),
        int(num_threads),
    )
    return tuple(_phase_c_run_payload_to_record(item) for item in raw_list)


def run_integrated_acb_batch(
    jobs: InstanceSpec,
    instance_id: str,
    replicates: Sequence[int],
    *,
    frozen_a_id: str,
    frozen_b_id: str,
    frozen_c_id: str,
    max_ls_evals: int = 50,
    evaluation_budget: int | None = None,
    stage_c_fraction: float = 0.85,
    num_threads: int = 0,
) -> tuple[FinalIntegratedRunRecord, ...]:
    """Run integrated `P5 (A* -> C* [85%] -> B* [15%])` across replicates in Rust."""
    raw = _native.run_integrated_acb_batch(
        _normalize_jobs(jobs),
        instance_id,
        [int(r) for r in replicates],
        frozen_a_id,
        frozen_b_id,
        frozen_c_id,
        int(max_ls_evals),
        (int(evaluation_budget) if evaluation_budget is not None else None),
        float(stage_c_fraction),
        int(num_threads),
    )
    return tuple(
        FinalIntegratedRunRecord(
            pipeline_id=str(pid),
            total_budget=int(tb),
            evals_used=int(eu),
            stage_c_best_makespan=int(cb),
            final_best_makespan=int(fb),
            stage_c_best_eval=int(cbe),
            final_best_eval=int(fbe),
            feasible_final=bool(feas),
            anytime_cmax_auc=float(auc),
            total_elapsed_ns=int(ns),
        )
        for pid, tb, eu, cb, fb, cbe, fbe, feas, auc, ns in raw
    )


@dataclass(frozen=True, slots=True)
class BranchLsTelemetryRecord:
    """Per-branch decode, local-search evaluation, and move-acceptance counters."""

    decode_evals: int
    ls_evals: int
    ls_calls: int
    improving_calls: int
    evaluated_moves: int
    feasible_moves: int
    rejected_cyclic_moves: int
    improving_moves: int
    accepted_moves: int
    rejected_moves: int


@dataclass(frozen=True, slots=True)
class DualViewExploratoryResultRecord:
    """Complete telemetry returned from one exploratory Dual-View LS budget run."""

    experiment_id: str
    base_config_id: str
    initializer_id: str
    ls_neighborhood: str
    ls_strategy: str
    max_ls_evals: int
    candidate_fraction: float
    genotype_capacity: int
    evaluation_budget: int
    evals_used: int
    init_evals: int
    search_evals: int
    generations_executed: int
    best_makespan: int
    best_found_eval: int
    best_found_generation: int
    feasible_final: bool
    anytime_cmax_auc: float
    total_elapsed_ns: int
    final_obr_diversity: float
    final_lam_diversity: float
    final_bald_diversity: float
    final_gate_p: float | None
    lam_best_makespan: int
    bald_best_makespan: int
    crossover_count: int
    mutation_count: int
    writeback_count: int
    migrations_applied: int
    lam_branch: BranchLsTelemetryRecord
    bald_branch: BranchLsTelemetryRecord
    gate_p_trajectory: tuple[tuple[int, float], ...]
    diversity_trajectory: tuple[tuple[int, float, float, float], ...]
    best_milestones: tuple[tuple[int, int], ...]
    best_chromosome: tuple[int, ...]
    best_machine_sequences: tuple[tuple[tuple[int, int], ...], ...]


def _branch_tuple_to_record(raw: tuple[int, ...]) -> BranchLsTelemetryRecord:
    return BranchLsTelemetryRecord(
        decode_evals=int(raw[0]),
        ls_evals=int(raw[1]),
        ls_calls=int(raw[2]),
        improving_calls=int(raw[3]),
        evaluated_moves=int(raw[4]),
        feasible_moves=int(raw[5]),
        rejected_cyclic_moves=int(raw[6]),
        improving_moves=int(raw[7]),
        accepted_moves=int(raw[8]),
        rejected_moves=int(raw[9]),
    )


def _exploratory_payload_to_record(payload: tuple[Any, ...]) -> DualViewExploratoryResultRecord:
    c, p, d, lam_b, bald_b, gate_t, div_t, milestones, best_chr, seqs = payload
    return DualViewExploratoryResultRecord(
        experiment_id=str(c[0]),
        base_config_id=str(c[1]),
        initializer_id=str(c[2]),
        ls_neighborhood=str(c[3]),
        ls_strategy=str(c[4]),
        max_ls_evals=int(c[5]),
        candidate_fraction=float(c[6]),
        genotype_capacity=int(c[7]),
        evaluation_budget=int(c[8]),
        evals_used=int(p[0]),
        init_evals=int(p[1]),
        search_evals=int(p[2]),
        generations_executed=int(p[3]),
        best_makespan=int(p[4]),
        best_found_eval=int(p[5]),
        best_found_generation=int(p[6]),
        feasible_final=bool(p[7]),
        anytime_cmax_auc=float(p[8]),
        total_elapsed_ns=int(p[9]),
        final_obr_diversity=float(d[0]),
        final_lam_diversity=float(d[1]),
        final_bald_diversity=float(d[2]),
        final_gate_p=(float(d[3]) if d[3] is not None else None),
        lam_best_makespan=int(d[4]),
        bald_best_makespan=int(d[5]),
        crossover_count=int(d[6]),
        mutation_count=int(d[7]),
        writeback_count=int(d[8]),
        migrations_applied=int(d[9]),
        lam_branch=_branch_tuple_to_record(lam_b),
        bald_branch=_branch_tuple_to_record(bald_b),
        gate_p_trajectory=tuple((int(g), float(gv)) for g, gv in gate_t),
        diversity_trajectory=tuple(
            (int(g), float(od), float(ld), float(bd)) for g, od, ld, bd in div_t
        ),
        best_milestones=tuple((int(ev), int(cm)) for ev, cm in milestones),
        best_chromosome=tuple(int(x) for x in best_chr),
        best_machine_sequences=tuple(tuple((int(j), int(op)) for j, op in ms) for ms in seqs),
    )


def run_dual_view_exploratory_single(
    jobs: InstanceSpec,
    experiment_id: str,
    instance_id: str,
    *,
    base_config_id: str = "C06",
    initializer_id: str = "A11",
    ls_neighborhood: str = "N5",
    ls_strategy: str = "steepest_descent",
    max_ls_evals: int = 50,
    candidate_fraction: float = 1.0,
    pop_size: int = 64,
    tournament_size: int = 2,
    crossover_rate: float = 0.85,
    mutation_rate: float = 0.15,
    elitism_count: int = 2,
    evaluation_budget: int | None = None,
    replicate_index: int = 0,
    master_seed_override: int | None = None,
    pairing_mode: str = "paired_init",
    pairing_group: str = "phase_c_shared_init",
    configuration_id: str | None = None,
) -> DualViewExploratoryResultRecord:
    """Execute one exploratory Dual-View LS budget allocation run in Rust."""
    payload = _native.run_dual_view_exploratory_single(
        _normalize_jobs(jobs),
        experiment_id,
        instance_id,
        base_config_id,
        initializer_id,
        ls_neighborhood,
        ls_strategy,
        int(max_ls_evals),
        float(candidate_fraction),
        int(pop_size),
        int(tournament_size),
        float(crossover_rate),
        float(mutation_rate),
        int(elitism_count),
        (int(evaluation_budget) if evaluation_budget is not None else None),
        int(replicate_index),
        master_seed_override,
        pairing_mode,
        pairing_group,
        configuration_id,
    )
    return _exploratory_payload_to_record(payload)


def run_dual_view_exploratory_batch(
    jobs: InstanceSpec,
    instance_id: str,
    tasks: Sequence[tuple[str, str, str, str, int, float, int, int | None, str, str, str]],
    *,
    initializer_id: str = "A11",
    pop_size: int = 64,
    tournament_size: int = 2,
    crossover_rate: float = 0.85,
    mutation_rate: float = 0.15,
    elitism_count: int = 2,
    evaluation_budget: int | None = None,
    num_threads: int = 0,
) -> tuple[DualViewExploratoryResultRecord, ...]:
    """Run a multi-threaded batch of exploratory Dual-View LS budget runs in Rust."""
    norm_tasks = [
        (
            str(exp_id),
            str(base_cfg),
            str(nb),
            str(strat),
            int(b_ls),
            float(frac),
            int(rep),
            seed_ov,
            str(pmode),
            str(pgroup),
            str(cfg_label),
        )
        for (
            exp_id,
            base_cfg,
            nb,
            strat,
            b_ls,
            frac,
            rep,
            seed_ov,
            pmode,
            pgroup,
            cfg_label,
        ) in tasks
    ]
    raw_list = _native.run_dual_view_exploratory_batch(
        _normalize_jobs(jobs),
        instance_id,
        norm_tasks,
        initializer_id,
        int(pop_size),
        int(tournament_size),
        float(crossover_rate),
        float(mutation_rate),
        int(elitism_count),
        (int(evaluation_budget) if evaluation_budget is not None else None),
        int(num_threads),
    )
    return tuple(_exploratory_payload_to_record(item) for item in raw_list)
