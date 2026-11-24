"""Benchmark track, split, provenance, and bound metadata definitions."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, model_validator

from jssp_research.benchmarks.checksums import is_valid_sha256_hex


class BenchmarkTrack(StrEnum):
    """Authoritative benchmark track identifiers from `RESEARCH_PLAN_V1.1.md`."""

    TRACK1_CORE_CLASSICAL = "track1_core_classical"
    TRACK2A_LARGE_TA = "track2a_large_ta"
    TRACK2B_STRESS = "track2b_stress"
    TRACK3_STRASSL_MUSLIU = "track3_strassl_musliu"
    TRACK4_VARIABLE_ROUTE = "track4_variable_route"
    TRACK5_BEL_INDUSTRIAL = "track5_bel_industrial"


class SplitAssignment(StrEnum):
    """Dataset partition role preventing tuning leakage onto held-out instances."""

    DEVELOPMENT = "development"
    CONFIRMATORY = "confirmatory"
    SCALABILITY_HELD_OUT = "scalability_held_out"
    STRESS_HELD_OUT = "stress_held_out"
    DEFERRED = "deferred"


class AcquisitionStatus(StrEnum):
    """Indicates whether raw benchmark files are committed or locally acquired and verified."""

    COMMITTED_FIXTURE = "committed_fixture"
    ACQUIRED_VERIFIED_LOCAL = "acquired_verified_local"
    METADATA_CATALOGED_FILE_PENDING = "metadata_cataloged_file_pending"
    DEFERRED_UNSUPPORTED_MODEL = "deferred_unsupported_model"


class BoundStatus(StrEnum):
    """Distinguishes proven optima from best-known upper bounds and unacquired snapshots."""

    PROVEN_OPTIMAL = "proven_optimal"
    BEST_KNOWN_OPEN = "best_known_open"
    PENDING_SNAPSHOT = "pending_snapshot"


class BoundProvenance(BaseModel):
    """Time-stamped provenance record for an instance's optimum or BKS/LB bounds."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: BoundStatus
    known_optimum: int | None = None
    best_known_upper_bound: int | None = None
    lower_bound: int | None = None
    source: str
    recorded_at: str

    @model_validator(mode="after")
    def _validate_bound_consistency(self) -> BoundProvenance:
        if self.status == BoundStatus.PROVEN_OPTIMAL and self.known_optimum is None:
            raise ValueError("PROVEN_OPTIMAL status requires a non-null known_optimum")
        if self.status != BoundStatus.PROVEN_OPTIMAL and self.known_optimum is not None:
            raise ValueError("known_optimum must only be set when status is PROVEN_OPTIMAL")
        if (
            self.lower_bound is not None
            and self.best_known_upper_bound is not None
            and self.lower_bound > self.best_known_upper_bound
        ):
            raise ValueError("lower_bound cannot exceed best_known_upper_bound")
        return self


class SourceProvenance(BaseModel):
    """Citation, source repository, license, and local availability metadata."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    citation: str
    source_url: str
    retrieval_date: str | None = None
    license_or_terms: str


class BenchmarkCatalogEntry(BaseModel):
    """Complete catalog and provenance entry for a single benchmark instance."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    instance_id: str
    family: str
    track: BenchmarkTrack
    status: AcquisitionStatus
    split: SplitAssignment
    num_jobs: int | None = None
    num_machines: int | None = None
    operation_count: int | None = None
    processing_time_distribution: str
    format_id: str | None = None
    parser_version: str | None = None
    local_relative_path: str | None = None
    raw_sha256: str | None = None
    canonical_sha256: str | None = None
    source_provenance: SourceProvenance
    bounds: BoundProvenance
    notes: str = ""

    @model_validator(mode="after")
    def _validate_entry_invariants(self) -> BenchmarkCatalogEntry:
        deferred_tracks = {
            BenchmarkTrack.TRACK4_VARIABLE_ROUTE,
            BenchmarkTrack.TRACK5_BEL_INDUSTRIAL,
        }
        if self.track in deferred_tracks:
            if self.status != AcquisitionStatus.DEFERRED_UNSUPPORTED_MODEL:
                raise ValueError(f"{self.track.value} entries must use DEFERRED_UNSUPPORTED_MODEL")
            if self.split != SplitAssignment.DEFERRED:
                raise ValueError(f"{self.track.value} entries must use SplitAssignment.DEFERRED")
            if self.format_id is not None or self.parser_version is not None:
                raise ValueError(f"Deferred {self.track.value} entries must not reference a parser")

        if self.num_jobs is not None and self.num_machines is not None:
            expected_ops = self.num_jobs * self.num_machines
            if self.operation_count is not None and self.operation_count != expected_ops:
                raise ValueError(
                    f"{self.instance_id}: operation_count {self.operation_count} != {expected_ops}"
                )

        if self.raw_sha256 is not None and not is_valid_sha256_hex(self.raw_sha256):
            raise ValueError(f"{self.instance_id}: invalid raw_sha256 hex digest")
        if self.canonical_sha256 is not None and not is_valid_sha256_hex(self.canonical_sha256):
            raise ValueError(f"{self.instance_id}: invalid canonical_sha256 hex digest")

        return self
