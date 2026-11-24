"""Immutable, deeply frozen, content-addressed run manifest schema and canonical serialization."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from jssp_research.benchmarks.checksums import is_valid_sha256_hex
from jssp_research.benchmarks.loaders import ParsedJsspInstance
from jssp_research.benchmarks.provenance import BenchmarkCatalogEntry, BenchmarkTrack
from jssp_research.experiments.budgets import BudgetSpec
from jssp_research.experiments.registry import (
    CANONICAL_DECODER_ID,
    CANONICAL_DECODER_VERSION,
    CANONICAL_OBJECTIVE_ID,
    CANONICAL_TIME_SEMANTICS,
    ExperimentPhase,
    ProtocolStage,
)
from jssp_research.experiments.seeds import (
    PairingMode,
    SeedContext,
    resolve_named_streams,
    rust_rng_metadata,
)
from jssp_research.native import get_frozen_master_seeds
from jssp_research.utils.provenance import (
    EnvironmentProvenance,
    GitProvenance,
    collect_environment_provenance,
    collect_git_provenance,
)

MANIFEST_SCHEMA_VERSION = "jssp-manifest-v1"
SUPPORTED_MANIFEST_VERSIONS: frozenset[str] = frozenset({MANIFEST_SCHEMA_VERSION})

CANONICAL_SERIALIZATION_VERSION = "jssp-canonical-json-v1"
SUPPORTED_CANONICAL_SERIALIZATION_VERSIONS: frozenset[str] = frozenset(
    {CANONICAL_SERIALIZATION_VERSION}
)

_I64_MIN = -(1 << 63)
_U64_MAX = (1 << 64) - 1


class ManifestIntegrityError(Exception):
    """Raised when manifest content hashes, canonical encoding, or schema versions fail."""


class ConfirmatoryIntegrityError(ManifestIntegrityError):
    """Raised when a confirmatory manifest violates strict cleanliness/freeze rules."""


def deep_freeze_container(value: Any) -> Any:
    """Recursively copy and convert nested mappings/lists into read-only structures."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        frozen_dict: dict[str, Any] = {}
        for key, item in sorted(value.items(), key=lambda kv: str(kv[0])):
            if not isinstance(key, str):
                raise ManifestIntegrityError(
                    f"Canonical manifest mappings require str keys, got {type(key).__name__}"
                )
            frozen_dict[key] = deep_freeze_container(item)
        return MappingProxyType(frozen_dict)
    if isinstance(value, (list, tuple)):
        return tuple(deep_freeze_container(item) for item in value)
    raise ManifestIntegrityError(
        f"Unsupported mutable or non-JSON container type {type(value).__name__} in manifest"
    )


def deep_thaw_container(value: Any) -> Any:
    """Convert frozen `MappingProxyType`/`tuple` trees back into plain JSON-serializable objects."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(k): deep_thaw_container(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [deep_thaw_container(item) for item in value]
    raise ManifestIntegrityError(
        f"Unexpected non-JSON value type {type(value).__name__} during serialization"
    )


def normalize_for_canonical_json(value: Any) -> Any:
    """Normalize data under `jssp-canonical-json-v1` determinism rules.

    Contract guarantees:
    1. Recursive mapping key ordering by Unicode code point (keys must be `str`).
    2. Strict UTF-8 string validation (rejects lone surrogates).
    3. Exact base-10 integer representation across full signed 64-bit / unsigned `u64` range
       (`[-(2^63), 2^64 - 1]`), avoiding IEEE-754 float coercion above `2^53 - 1`.
    4. Finite floats only (`NaN`, `+Infinity`, `-Infinity` rejected).
    5. Negative zero (`-0.0`) normalized to `+0.0`.
    """
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value < _I64_MIN or value > _U64_MAX:
            raise ManifestIntegrityError(
                f"Integer {value} exceeds canonical i64/u64 bounds [{_I64_MIN}, {_U64_MAX}]"
            )
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ManifestIntegrityError(
                f"Non-finite float value {value!r} is forbidden in canonical JSON"
            )
        return 0.0 if value == 0.0 else value
    if isinstance(value, str):
        try:
            value.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ManifestIntegrityError(
                f"String contains invalid UTF-8 / lone surrogate code point: {exc}"
            ) from exc
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key in sorted(value.keys(), key=lambda k: str(k)):
            if not isinstance(key, str):
                raise ManifestIntegrityError(
                    f"Canonical JSON object keys must be str, got {type(key).__name__}"
                )
            normalize_for_canonical_json(key)
            normalized[key] = normalize_for_canonical_json(value[key])
        return normalized
    if isinstance(value, (list, tuple)):
        return [normalize_for_canonical_json(item) for item in value]
    raise ManifestIntegrityError(
        f"Unsupported value type {type(value).__name__} in canonical JSON payload"
    )


def canonical_json_bytes(
    payload: Mapping[str, Any],
    *,
    serialization_version: str = CANONICAL_SERIALIZATION_VERSION,
) -> bytes:
    """Serialize `payload` into deterministic canonical UTF-8 JSON bytes."""
    if serialization_version not in SUPPORTED_CANONICAL_SERIALIZATION_VERSIONS:
        raise ManifestIntegrityError(
            f"Unsupported canonical serialization version {serialization_version!r}"
        )
    normalized = normalize_for_canonical_json(payload)
    encoded = json.dumps(
        normalized,
        separators=(",", ":"),
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    )
    return encoded.encode("utf-8")


def canonical_json_sha256(
    payload: Mapping[str, Any],
    *,
    serialization_version: str = CANONICAL_SERIALIZATION_VERSION,
) -> str:
    """Compute SHA-256 hex digest of `canonical_json_bytes(payload)`."""
    return hashlib.sha256(
        canonical_json_bytes(payload, serialization_version=serialization_version)
    ).hexdigest()


class ManifestIdentitySection(BaseModel):
    """Unique identification and lifecycle stage for an intended experiment run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str = MANIFEST_SCHEMA_VERSION
    run_id: str
    experiment_id: str
    phase: ExperimentPhase
    protocol_stage: ProtocolStage
    configuration_id: str
    replicate_index: int
    created_at_utc: str

    @model_validator(mode="after")
    def _validate_identity(self) -> ManifestIdentitySection:
        if self.schema_version not in SUPPORTED_MANIFEST_VERSIONS:
            raise ManifestIntegrityError(
                f"Unsupported manifest schema version {self.schema_version!r}"
            )
        for field_name in ("run_id", "experiment_id", "configuration_id"):
            if not getattr(self, field_name):
                raise ValueError(f"{field_name} must not be empty")
        if self.replicate_index < 0:
            raise ValueError("replicate_index must be >= 0")
        datetime.fromisoformat(self.created_at_utc)
        return self


class ManifestAlgorithmSection(BaseModel):
    """Declared algorithm, neighborhood, and deeply immutable parameter snapshot."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    algorithm_id: str
    algorithm_version: str
    neighborhood_id: str | None = None
    configuration_snapshot: Mapping[str, Any] = Field(default_factory=lambda: MappingProxyType({}))
    canonical_configuration_sha256: str

    @field_validator("configuration_snapshot", mode="after")
    @classmethod
    def _freeze_snapshot(cls, value: Mapping[str, Any]) -> Mapping[str, Any]:
        return deep_freeze_container(value)

    @field_serializer("configuration_snapshot")
    def _serialize_snapshot(self, value: Mapping[str, Any]) -> Any:
        return deep_thaw_container(value)

    @model_validator(mode="after")
    def _validate_algorithm(self) -> ManifestAlgorithmSection:
        if not self.algorithm_id or not self.algorithm_version:
            raise ValueError("algorithm_id and algorithm_version must be non-empty")
        expected_hash = canonical_json_sha256(self.configuration_snapshot)
        if self.canonical_configuration_sha256 != expected_hash:
            raise ManifestIntegrityError(
                f"canonical_configuration_sha256 mismatch: expected {expected_hash}, "
                f"got {self.canonical_configuration_sha256}"
            )
        return self


class ManifestInstanceSection(BaseModel):
    """Exact benchmark instance, parser, dimension, and SHA-256 provenance."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    benchmark_track: BenchmarkTrack
    family: str
    instance_id: str
    input_sha256: str
    canonical_parsed_sha256: str
    parser_name: str
    parser_version: str
    num_jobs: int
    num_machines: int
    operation_count: int

    @model_validator(mode="after")
    def _validate_instance_section(self) -> ManifestInstanceSection:
        if not is_valid_sha256_hex(self.input_sha256):
            raise ManifestIntegrityError(f"Invalid input_sha256: {self.input_sha256!r}")
        if not is_valid_sha256_hex(self.canonical_parsed_sha256):
            raise ManifestIntegrityError(
                f"Invalid canonical_parsed_sha256: {self.canonical_parsed_sha256!r}"
            )
        if self.num_jobs <= 0 or self.num_machines <= 0:
            raise ValueError("Instance dimensions must be positive")
        if self.operation_count != self.num_jobs * self.num_machines:
            raise ValueError("operation_count must equal num_jobs * num_machines")
        return self


class ManifestObjectiveSection(BaseModel):
    """Fixed classical JSSP makespan objective and Serial SGS decoder contract."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    objective_id: str = CANONICAL_OBJECTIVE_ID
    time_semantics: str = CANONICAL_TIME_SEMANTICS
    decoder_id: str = CANONICAL_DECODER_ID
    decoder_version: str = CANONICAL_DECODER_VERSION


class NamedStreamRecord(BaseModel):
    """Derived 256-bit seed fingerprint and coupling tag for one D07-A stream."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: str
    coupling_key: str
    derived_seed_sha256: str

    @model_validator(mode="after")
    def _validate_seed_hex(self) -> NamedStreamRecord:
        if not is_valid_sha256_hex(self.derived_seed_sha256):
            raise ManifestIntegrityError(f"Stream {self.role!r} seed hash is not valid 64-char hex")
        return self


class ManifestRandomnessSection(BaseModel):
    """Complete deterministic PRNG algorithm, derivation frame, and stream hashes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    master_seed: int
    replicate_index: int
    master_seed_schedule_id: str
    rng_algorithm: str
    rng_crate_version: str
    hash_crate_version: str
    seed_derivation_schema: str
    pairing_policy: PairingMode
    pairing_group: str
    streams: tuple[NamedStreamRecord, ...]
    qmc_scramble_metadata: Mapping[str, Any] | None = None

    @field_validator("qmc_scramble_metadata", mode="after")
    @classmethod
    def _freeze_qmc_meta(cls, value: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
        if value is None:
            return None
        return deep_freeze_container(value)

    @field_serializer("qmc_scramble_metadata")
    def _serialize_qmc_meta(self, value: Mapping[str, Any] | None) -> Any:
        if value is None:
            return None
        return deep_thaw_container(value)

    @model_validator(mode="after")
    def _validate_randomness(self) -> ManifestRandomnessSection:
        if self.replicate_index < 0:
            raise ValueError("randomness.replicate_index must be >= 0")
        if self.master_seed < 0 or self.master_seed >= (1 << 64):
            raise ValueError("randomness.master_seed must be an unsigned 64-bit integer")
        if len(self.streams) != 9:
            raise ManifestIntegrityError(
                f"Expected exactly 9 D07-A named streams, found {len(self.streams)}"
            )
        roles = [s.role for s in self.streams]
        if len(set(roles)) != 9:
            raise ManifestIntegrityError("Duplicate stream roles in manifest randomness section")
        return self


class ManifestSourceCodeSection(BaseModel):
    """Git commit/diff state plus Python framework package version."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    git: GitProvenance
    package_version: str


class RunManifest(BaseModel):
    """Deeply immutable, self-verifying specification manifest for a single experiment run.

    Legacy pre-M3.1 manifests without `canonical_serialization_version` remain
    compatible under `jssp-canonical-json-v1` hashing without mutating stored hashes.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    identity: ManifestIdentitySection
    source_code: ManifestSourceCodeSection
    algorithm: ManifestAlgorithmSection
    instance: ManifestInstanceSection
    objective: ManifestObjectiveSection
    randomness: ManifestRandomnessSection
    budget: BudgetSpec
    environment: EnvironmentProvenance
    canonical_serialization_version: str | None = None
    manifest_sha256: str

    def body_dict(self) -> dict[str, Any]:
        """Serialize all manifest sections excluding `manifest_sha256`.

        Preserves legacy 8-section hash compatibility when `canonical_serialization_version`
        is absent (`None`).
        """
        exclude_fields = {"manifest_sha256"}
        if self.canonical_serialization_version is None:
            exclude_fields.add("canonical_serialization_version")
        return self.model_dump(mode="json", exclude=exclude_fields)

    def effective_serialization_version(self) -> str:
        """Return explicit canonical serialization profile or default compatibility version."""
        return self.canonical_serialization_version or CANONICAL_SERIALIZATION_VERSION

    def verify_integrity(self) -> None:
        """Recompute configuration and manifest hashes from live values and verify invariants."""
        ser_ver = self.effective_serialization_version()
        if ser_ver not in SUPPORTED_CANONICAL_SERIALIZATION_VERSIONS:
            raise ManifestIntegrityError(f"Unsupported canonical serialization version {ser_ver!r}")

        expected_cfg = canonical_json_sha256(
            self.algorithm.configuration_snapshot,
            serialization_version=ser_ver,
        )
        if self.algorithm.canonical_configuration_sha256 != expected_cfg:
            raise ManifestIntegrityError(
                f"canonical_configuration_sha256 mismatch: expected {expected_cfg}, "
                f"got {self.algorithm.canonical_configuration_sha256}"
            )

        expected_sha = canonical_json_sha256(self.body_dict(), serialization_version=ser_ver)
        if self.manifest_sha256 != expected_sha:
            raise ManifestIntegrityError(
                f"manifest_sha256 mismatch: expected {expected_sha}, got {self.manifest_sha256}"
            )

        if self.identity.protocol_stage == ProtocolStage.CONFIRMATORY:
            validate_confirmatory_manifest(self)

    @model_validator(mode="after")
    def _validate_manifest_invariants(self) -> RunManifest:
        if self.identity.replicate_index != self.randomness.replicate_index:
            raise ManifestIntegrityError(
                "identity.replicate_index does not match randomness.replicate_index"
            )
        self.verify_integrity()
        return self


def validate_confirmatory_manifest(manifest: RunManifest) -> None:
    """Enforce strict frozen-state integrity rules for confirmatory experiments."""
    git_info = manifest.source_code.git
    if not git_info.commit:
        raise ConfirmatoryIntegrityError("Confirmatory runs require a valid Git commit SHA")
    if git_info.dirty is not False or git_info.diff_sha256 is not None:
        raise ConfirmatoryIntegrityError(
            "Confirmatory runs forbid a dirty Git working tree or non-null diff hash"
        )

    env = manifest.environment
    if not env.cargo_lock_sha256 or not is_valid_sha256_hex(env.cargo_lock_sha256):
        raise ConfirmatoryIntegrityError("Confirmatory runs require a valid Cargo.lock SHA-256")
    if not env.uv_lock_sha256 or not is_valid_sha256_hex(env.uv_lock_sha256):
        raise ConfirmatoryIntegrityError("Confirmatory runs require a valid uv.lock SHA-256")

    frozen_seeds = get_frozen_master_seeds()
    rep = manifest.randomness.replicate_index
    if rep < 0 or rep >= len(frozen_seeds):
        raise ConfirmatoryIntegrityError(
            f"Confirmatory replicate_index {rep} outside frozen schedule 0..{len(frozen_seeds)}"
        )
    if manifest.randomness.master_seed != frozen_seeds[rep]:
        raise ConfirmatoryIntegrityError(
            f"Confirmatory master_seed {manifest.randomness.master_seed} does not match "
            f"frozen schedule value {frozen_seeds[rep]} for replicate {rep}"
        )

    if manifest.instance.benchmark_track in {
        BenchmarkTrack.TRACK4_VARIABLE_ROUTE,
        BenchmarkTrack.TRACK5_BEL_INDUSTRIAL,
    }:
        raise ConfirmatoryIntegrityError(
            f"Deferred benchmark track {manifest.instance.benchmark_track.value} "
            "cannot be used in confirmatory runs"
        )


def create_run_manifest(
    *,
    run_id: str,
    experiment_id: str,
    phase: ExperimentPhase | str,
    protocol_stage: ProtocolStage | str,
    configuration_id: str,
    algorithm_id: str,
    algorithm_version: str,
    neighborhood_id: str | None,
    configuration_snapshot: Mapping[str, Any],
    catalog_entry: BenchmarkCatalogEntry,
    parsed_instance: ParsedJsspInstance,
    seed_context: SeedContext,
    budget: BudgetSpec,
    repo_root: Path,
    package_version: str = "0.1.0",
    qmc_scramble_metadata: Mapping[str, Any] | None = None,
    compiler_profile: str = "dev_editable",
) -> RunManifest:
    """Assemble and content-hash a fresh deeply immutable `RunManifest` before run execution."""
    stage = ProtocolStage(protocol_stage)
    phase_enum = ExperimentPhase(phase)

    git_prov = collect_git_provenance(repo_root)
    env_prov = collect_environment_provenance(repo_root, compiler_profile=compiler_profile)
    rng_meta = rust_rng_metadata()
    stream_previews = resolve_named_streams(seed_context, preview_count=0)

    frozen_snapshot = deep_freeze_container(configuration_snapshot)
    frozen_qmc_meta = (
        deep_freeze_container(qmc_scramble_metadata) if qmc_scramble_metadata is not None else None
    )

    identity = ManifestIdentitySection(
        schema_version=MANIFEST_SCHEMA_VERSION,
        run_id=run_id,
        experiment_id=experiment_id,
        phase=phase_enum,
        protocol_stage=stage,
        configuration_id=configuration_id,
        replicate_index=seed_context.replicate_index,
        created_at_utc=datetime.now(tz=UTC).isoformat(),
    )
    source_code = ManifestSourceCodeSection(git=git_prov, package_version=package_version)
    algorithm = ManifestAlgorithmSection(
        algorithm_id=algorithm_id,
        algorithm_version=algorithm_version,
        neighborhood_id=neighborhood_id,
        configuration_snapshot=frozen_snapshot,
        canonical_configuration_sha256=canonical_json_sha256(frozen_snapshot),
    )
    instance = ManifestInstanceSection(
        benchmark_track=catalog_entry.track,
        family=catalog_entry.family,
        instance_id=parsed_instance.instance_id,
        input_sha256=parsed_instance.raw_sha256,
        canonical_parsed_sha256=parsed_instance.canonical_sha256,
        parser_name=parsed_instance.format_id,
        parser_version=parsed_instance.parser_version,
        num_jobs=parsed_instance.num_jobs,
        num_machines=parsed_instance.num_machines,
        operation_count=parsed_instance.num_operations,
    )
    objective = ManifestObjectiveSection()
    randomness = ManifestRandomnessSection(
        master_seed=seed_context.master_seed,
        replicate_index=seed_context.replicate_index,
        master_seed_schedule_id=rng_meta.master_seed_schedule_id,
        rng_algorithm=rng_meta.algorithm,
        rng_crate_version=rng_meta.rng_crate_version,
        hash_crate_version=rng_meta.hash_crate_version,
        seed_derivation_schema=rng_meta.seed_derivation_schema,
        pairing_policy=seed_context.pairing_mode,
        pairing_group=seed_context.pairing_group,
        streams=tuple(
            NamedStreamRecord(
                role=row.role,
                coupling_key=row.coupling_key,
                derived_seed_sha256=row.seed_hex,
            )
            for row in stream_previews
        ),
        qmc_scramble_metadata=frozen_qmc_meta,
    )

    body = {
        "canonical_serialization_version": CANONICAL_SERIALIZATION_VERSION,
        "identity": identity.model_dump(mode="json"),
        "source_code": source_code.model_dump(mode="json"),
        "algorithm": algorithm.model_dump(mode="json"),
        "instance": instance.model_dump(mode="json"),
        "objective": objective.model_dump(mode="json"),
        "randomness": randomness.model_dump(mode="json"),
        "budget": budget.model_dump(mode="json"),
        "environment": env_prov.model_dump(mode="json"),
    }
    manifest_sha256 = canonical_json_sha256(body)

    return RunManifest(
        identity=identity,
        source_code=source_code,
        algorithm=algorithm,
        instance=instance,
        objective=objective,
        randomness=randomness,
        budget=budget,
        environment=env_prov,
        canonical_serialization_version=CANONICAL_SERIALIZATION_VERSION,
        manifest_sha256=manifest_sha256,
    )


def write_manifest(manifest: RunManifest, path: Path) -> Path:
    """Persist `manifest` atomically as canonical JSON, refusing to overwrite existing files."""
    if path.exists():
        raise FileExistsError(f"Manifest path already exists and cannot be overwritten: {path}")
    manifest.verify_integrity()
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    path.write_text(serialized, encoding="utf-8")
    return path


def load_and_validate_manifest(path: Path) -> RunManifest:
    """Read, deserialize, and verify canonical hash and confirmatory rules for a manifest file."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    return RunManifest.model_validate(raw)
