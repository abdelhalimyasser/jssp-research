"""Separately stored, versioned execution result records linked to immutable manifests."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator

from jssp_research.benchmarks.checksums import is_valid_sha256_hex
from jssp_research.experiments.manifests import RunManifest, canonical_json_sha256

RESULT_SCHEMA_VERSION = "jssp-result-v1"
SUPPORTED_RESULT_VERSIONS: frozenset[str] = frozenset({RESULT_SCHEMA_VERSION})


class ResultValidationError(Exception):
    """Raised when a run result record fails integrity or manifest cross-reference checks."""


class RunStatus(StrEnum):
    """Execution outcome status for an individual run attempt."""

    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"
    BUDGET_EXHAUSTED = "budget_exhausted"


class ExecutionMetricsSection(BaseModel):
    """Timing and evaluation-effort metrics persisted for one attempt."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    start_utc: str
    end_utc: str
    elapsed_seconds: float
    objective_evaluations_used: int
    local_search_evaluations_used: int = 0
    generations_or_iterations: int | None = None

    @model_validator(mode="after")
    def _validate_metrics(self) -> ExecutionMetricsSection:
        start_dt = datetime.fromisoformat(self.start_utc)
        end_dt = datetime.fromisoformat(self.end_utc)
        if end_dt < start_dt:
            raise ValueError("end_utc cannot precede start_utc")
        if self.elapsed_seconds < 0.0:
            raise ValueError("elapsed_seconds must be non-negative")
        if self.objective_evaluations_used < 0:
            raise ValueError("objective_evaluations_used must be non-negative")
        if self.local_search_evaluations_used < 0:
            raise ValueError("local_search_evaluations_used must be non-negative")
        return self


class ObjectiveResultSection(BaseModel):
    """Makespan, feasibility, and termination outcome recorded separately from manifests."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    makespan: int | None = None
    feasible: bool | None = None
    rpd_bks: float | None = None
    gap_lb: float | None = None
    reached_local_optimum: bool | None = None
    terminated_reason: str | None = None


class RunErrorInfo(BaseModel):
    """Typed diagnostic payload stored when a run attempt fails or aborts."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    error_type: str
    message: str
    stage: str


class RunResultRecord(BaseModel):
    """Immutable per-attempt execution record referencing an immutable `RunManifest`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str = RESULT_SCHEMA_VERSION
    run_id: str
    manifest_sha256: str
    attempt_index: int
    experiment_id: str
    configuration_id: str
    instance_id: str
    replicate_index: int
    master_seed: int
    status: RunStatus
    execution_metrics: ExecutionMetricsSection
    objective_results: ObjectiveResultSection
    error_info: RunErrorInfo | None = None
    artifact_refs: tuple[str, ...] = ()
    result_sha256: str

    def body_dict(self) -> dict[str, Any]:
        """Serialize all fields excluding `result_sha256` for canonical hashing."""
        return self.model_dump(mode="json", exclude={"result_sha256"})

    @model_validator(mode="after")
    def _validate_record(self) -> RunResultRecord:
        if self.schema_version not in SUPPORTED_RESULT_VERSIONS:
            raise ResultValidationError(
                f"Unsupported result schema version {self.schema_version!r}"
            )
        if not is_valid_sha256_hex(self.manifest_sha256):
            raise ResultValidationError(f"Invalid manifest_sha256: {self.manifest_sha256!r}")
        if self.attempt_index < 1:
            raise ValueError("attempt_index must be >= 1")
        if self.replicate_index < 0:
            raise ValueError("replicate_index must be >= 0")
        if self.master_seed < 0 or self.master_seed >= (1 << 64):
            raise ValueError("master_seed must be an unsigned 64-bit integer")

        if self.status == RunStatus.FAILED and self.error_info is None:
            raise ValueError("error_info is required when status is FAILED")
        if self.status == RunStatus.COMPLETED and self.objective_results.makespan is None:
            raise ValueError("COMPLETED result records require makespan")

        expected_sha = canonical_json_sha256(self.body_dict())
        if self.result_sha256 != expected_sha:
            raise ResultValidationError(
                f"result_sha256 mismatch: expected {expected_sha}, got {self.result_sha256}"
            )
        return self


def create_run_result_record(
    *,
    manifest: RunManifest,
    attempt_index: int,
    status: RunStatus | str,
    start_utc: datetime | str | None = None,
    end_utc: datetime | str | None = None,
    elapsed_seconds: float,
    objective_evaluations_used: int,
    local_search_evaluations_used: int = 0,
    generations_or_iterations: int | None = None,
    makespan: int | None = None,
    feasible: bool | None = None,
    rpd_bks: float | None = None,
    gap_lb: float | None = None,
    reached_local_optimum: bool | None = None,
    terminated_reason: str | None = None,
    error_info: RunErrorInfo | dict[str, str] | None = None,
    artifact_refs: tuple[str, ...] = (),
) -> RunResultRecord:
    """Create and content-hash a separate `RunResultRecord` bound to `manifest`."""
    now_iso = datetime.now(tz=UTC).isoformat()
    start_str = start_utc.isoformat() if isinstance(start_utc, datetime) else (start_utc or now_iso)
    end_str = end_utc.isoformat() if isinstance(end_utc, datetime) else (end_utc or now_iso)
    resolved_error = RunErrorInfo.model_validate(error_info) if error_info is not None else None

    exec_metrics = ExecutionMetricsSection(
        start_utc=start_str,
        end_utc=end_str,
        elapsed_seconds=elapsed_seconds,
        objective_evaluations_used=objective_evaluations_used,
        local_search_evaluations_used=local_search_evaluations_used,
        generations_or_iterations=generations_or_iterations,
    )
    obj_results = ObjectiveResultSection(
        makespan=makespan,
        feasible=feasible,
        rpd_bks=rpd_bks,
        gap_lb=gap_lb,
        reached_local_optimum=reached_local_optimum,
        terminated_reason=terminated_reason,
    )

    body = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "run_id": manifest.identity.run_id,
        "manifest_sha256": manifest.manifest_sha256,
        "attempt_index": attempt_index,
        "experiment_id": manifest.identity.experiment_id,
        "configuration_id": manifest.identity.configuration_id,
        "instance_id": manifest.instance.instance_id,
        "replicate_index": manifest.randomness.replicate_index,
        "master_seed": manifest.randomness.master_seed,
        "status": RunStatus(status).value,
        "execution_metrics": exec_metrics.model_dump(mode="json"),
        "objective_results": obj_results.model_dump(mode="json"),
        "error_info": resolved_error.model_dump(mode="json")
        if resolved_error is not None
        else None,
        "artifact_refs": list(artifact_refs),
    }
    result_sha = canonical_json_sha256(body)

    return RunResultRecord(
        schema_version=RESULT_SCHEMA_VERSION,
        run_id=manifest.identity.run_id,
        manifest_sha256=manifest.manifest_sha256,
        attempt_index=attempt_index,
        experiment_id=manifest.identity.experiment_id,
        configuration_id=manifest.identity.configuration_id,
        instance_id=manifest.instance.instance_id,
        replicate_index=manifest.randomness.replicate_index,
        master_seed=manifest.randomness.master_seed,
        status=RunStatus(status),
        execution_metrics=exec_metrics,
        objective_results=obj_results,
        error_info=resolved_error,
        artifact_refs=artifact_refs,
        result_sha256=result_sha,
    )


def validate_result_against_manifest(manifest: RunManifest, result: RunResultRecord) -> None:
    """Verify exact identity, seed, and evaluation-budget consistency across manifest and result."""
    checks = [
        ("run_id", result.run_id, manifest.identity.run_id),
        ("manifest_sha256", result.manifest_sha256, manifest.manifest_sha256),
        ("experiment_id", result.experiment_id, manifest.identity.experiment_id),
        ("configuration_id", result.configuration_id, manifest.identity.configuration_id),
        ("instance_id", result.instance_id, manifest.instance.instance_id),
        ("replicate_index", result.replicate_index, manifest.randomness.replicate_index),
        ("master_seed", result.master_seed, manifest.randomness.master_seed),
    ]
    for label, got, expected in checks:
        if got != expected:
            raise ResultValidationError(
                f"Result/Manifest mismatch for {label}: expected {expected!r}, got {got!r}"
            )

    budget_cap = manifest.budget.objective_evaluation_budget
    if result.execution_metrics.objective_evaluations_used > budget_cap:
        raise ResultValidationError(
            f"Evaluations used ({result.execution_metrics.objective_evaluations_used}) "
            f"exceed manifest budget ({budget_cap})"
        )
