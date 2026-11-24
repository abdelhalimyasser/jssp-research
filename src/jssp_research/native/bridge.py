"""Thin typed Python wrapper around the compiled `jssp_research._native` extension."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

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
