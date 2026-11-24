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
