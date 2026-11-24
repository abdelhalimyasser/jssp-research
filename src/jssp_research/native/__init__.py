"""Python-Rust native extension boundary package."""

from __future__ import annotations

from jssp_research.native.bridge import (
    DecodedScheduleResult,
    DerivedStreamPreview,
    RngMetadataRecord,
    ScheduledOperationRecord,
    decode_obr,
    decode_obr_append_only,
    derive_rng_streams,
    derive_rng_streams_with_seed,
    evaluate_makespan,
    get_bridge_info,
    get_core_version,
    get_frozen_master_seeds,
    get_native_bridge_info,
    get_rng_metadata,
    resolve_obr_operations,
    rng_sample_range_and_shuffle,
    validate_instance,
    validate_obr,
    validate_raw_schedule,
    validate_schedule,
)

__all__ = [
    "DecodedScheduleResult",
    "DerivedStreamPreview",
    "RngMetadataRecord",
    "ScheduledOperationRecord",
    "decode_obr",
    "decode_obr_append_only",
    "derive_rng_streams",
    "derive_rng_streams_with_seed",
    "evaluate_makespan",
    "get_bridge_info",
    "get_core_version",
    "get_frozen_master_seeds",
    "get_native_bridge_info",
    "get_rng_metadata",
    "resolve_obr_operations",
    "rng_sample_range_and_shuffle",
    "validate_instance",
    "validate_obr",
    "validate_raw_schedule",
    "validate_schedule",
]
