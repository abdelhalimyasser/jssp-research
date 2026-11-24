"""Deterministic seed schedule, named stream derivation, and pairing helpers."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, model_validator

from jssp_research.native import (
    DerivedStreamPreview,
    RngMetadataRecord,
    derive_rng_streams,
    derive_rng_streams_with_seed,
    get_frozen_master_seeds,
    get_rng_metadata,
)


class PairingMode(StrEnum):
    """Seed-coupling mode governing shared vs configuration-specific substreams.

    Note: `SHARED_STREAM_SEEDS` derives identical initial ChaCha20 seeds for all 9
    stream roles within a `pairing_group`, which is **shared stream seeds** rather than
    true trajectory-wide event-aligned Common Random Numbers (CRN). Only `paired_init`
    guarantees event alignment on fixed-draw initialization/scrambling stages.
    """

    SHARED_STREAM_SEEDS = "shared_stream_seeds"
    PAIRED_INIT = "paired_init"
    CONFIG_SCOPED = "config_scoped"
    # Backward-compatible enum alias mapped to the accurate policy name.
    CRN = "shared_stream_seeds"

    @classmethod
    def _missing_(cls, value: object) -> PairingMode | None:
        if value == "crn":
            return cls.SHARED_STREAM_SEEDS
        return None


class SeedContext(BaseModel):
    """Frozen seed-derivation context passed across manifest creation and Rust RNG calls."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    replicate_index: int
    master_seed: int
    instance_id: str
    pairing_mode: PairingMode = PairingMode.PAIRED_INIT
    pairing_group: str = ""
    configuration_id: str = ""
    is_frozen_confirmatory_seed: bool = True

    @model_validator(mode="after")
    def _validate_seed_context(self) -> SeedContext:
        if self.replicate_index < 0:
            raise ValueError("replicate_index must be non-negative")
        if not self.instance_id:
            raise ValueError("instance_id must not be empty")
        if self.master_seed < 0 or self.master_seed >= (1 << 64):
            raise ValueError("master_seed must be an unsigned 64-bit integer")
        if (
            self.pairing_mode in {PairingMode.SHARED_STREAM_SEEDS, PairingMode.PAIRED_INIT}
            and not self.pairing_group
        ):
            raise ValueError(f"pairing_group must be non-empty for mode {self.pairing_mode.value}")
        if (
            self.pairing_mode in {PairingMode.PAIRED_INIT, PairingMode.CONFIG_SCOPED}
            and not self.configuration_id
        ):
            raise ValueError(
                f"configuration_id must be non-empty for mode {self.pairing_mode.value}"
            )
        return self


def frozen_master_seed_for_replicate(replicate_index: int) -> int:
    """Return the frozen SHA-256 master seed for `replicate_index in 0..20`."""
    seeds = get_frozen_master_seeds()
    if replicate_index < 0 or replicate_index >= len(seeds):
        raise ValueError(
            f"replicate_index {replicate_index} is outside frozen schedule 0..{len(seeds)}"
        )
    return seeds[replicate_index]


def build_seed_context(
    *,
    replicate_index: int,
    instance_id: str,
    pairing_mode: PairingMode | str = PairingMode.PAIRED_INIT,
    pairing_group: str = "default_group",
    configuration_id: str = "default_config",
    explicit_master_seed: int | None = None,
) -> SeedContext:
    """Build a validated `SeedContext` using either frozen replicate seeds or an explicit seed."""
    mode = PairingMode(pairing_mode)
    frozen_seeds = get_frozen_master_seeds()
    if explicit_master_seed is None:
        master_seed = frozen_master_seed_for_replicate(replicate_index)
        is_frozen = True
    else:
        master_seed = int(explicit_master_seed)
        is_frozen = (
            0 <= replicate_index < len(frozen_seeds)
            and frozen_seeds[replicate_index] == master_seed
        )

    return SeedContext(
        replicate_index=replicate_index,
        master_seed=master_seed,
        instance_id=instance_id,
        pairing_mode=mode,
        pairing_group=pairing_group if mode != PairingMode.CONFIG_SCOPED else "",
        configuration_id=configuration_id if mode != PairingMode.SHARED_STREAM_SEEDS else "",
        is_frozen_confirmatory_seed=is_frozen,
    )


def resolve_named_streams(
    context: SeedContext,
    *,
    preview_count: int = 0,
) -> tuple[DerivedStreamPreview, ...]:
    """Derive all nine D07-A streams in Rust directly from `context`."""
    if context.is_frozen_confirmatory_seed and 0 <= context.replicate_index < 20:
        return derive_rng_streams(
            context.replicate_index,
            context.instance_id,
            context.pairing_mode.value,
            context.pairing_group,
            context.configuration_id,
            preview_count,
        )
    return derive_rng_streams_with_seed(
        context.master_seed,
        context.replicate_index,
        context.instance_id,
        context.pairing_mode.value,
        context.pairing_group,
        context.configuration_id,
        preview_count,
    )


def rust_rng_metadata() -> RngMetadataRecord:
    """Return authoritative RNG engine and schema metadata from Rust."""
    return get_rng_metadata()
