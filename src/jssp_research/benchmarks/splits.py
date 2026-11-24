"""Frozen stratified development and held-out split definitions (`D07-B`, `M3.1 E04`)."""

from __future__ import annotations

from dataclasses import dataclass

from jssp_research.benchmarks.provenance import SplitAssignment

# Stratified 48-instance Track 1 development split (~20% per family, across every family)
# matching `docs/experiments/fairness_protocol.md` and `RESEARCH_PLAN_V1.1.md`:
# FT:1/3, LA:8/40, ABZ:1/5, ORB:2/10, SWV:4/20, YN:1/4, TA:16/80, DMU:15/80 -> 48 dev / 194 test.
TRACK1_DEVELOPMENT_IDS: frozenset[str] = frozenset(
    {
        # FT (1)
        "ft06",
        # LA (8: one from each 5-instance size stratum)
        "la01",
        "la06",
        "la11",
        "la16",
        "la21",
        "la26",
        "la31",
        "la36",
        # ABZ (1)
        "abz5",
        # ORB (2)
        "orb01",
        "orb06",
        # SWV (4: one from each 5-instance stratum)
        "swv01",
        "swv06",
        "swv11",
        "swv16",
        # YN (1)
        "yn1",
        # TA (16: two from each 10-instance size stratum)
        "ta01",
        "ta06",
        "ta11",
        "ta16",
        "ta21",
        "ta26",
        "ta31",
        "ta36",
        "ta41",
        "ta46",
        "ta51",
        "ta56",
        "ta61",
        "ta66",
        "ta71",
        "ta76",
        # DMU (15: balanced across the 8 size strata)
        "dmu01",
        "dmu06",
        "dmu11",
        "dmu16",
        "dmu21",
        "dmu26",
        "dmu31",
        "dmu36",
        "dmu41",
        "dmu46",
        "dmu51",
        "dmu56",
        "dmu61",
        "dmu66",
        "dmu71",
    }
)

# ---------------------------------------------------------------------------
# Track 3 (Strassl & Musliu 2022) Pre-Registered Sample & Split Provenance
# ---------------------------------------------------------------------------

TRACK3_DATASET_ID = "strassl_musliu_2022_jssp_4225"
TRACK3_DATASET_VERSION = "v1 (Computers & Operations Research 141:105661, 2022)"
TRACK3_SOURCE_ARCHIVE_SHA256: str | None = None
TRACK3_SOURCE_VERIFICATION_STATUS = "pending_raw_archive_acquisition"
TRACK3_SELECTION_METHOD = "stratified_16_cell_equal_allocation_10_per_stratum"
TRACK3_SELECTION_SEED = "jssp-research-v1-track3-selection-seed (0x5433534D32303236)"
TRACK3_PRE_OUTCOME_BLINDED = True


@dataclass(frozen=True, slots=True)
class Track3StratumSpec:
    """Feature-stratum definition governing Track 3 Strassl-Musliu sampling."""

    stratum_id: str
    size_regime: str
    job_range: tuple[int, int]
    machine_range: tuple[int, int]
    aspect_ratio_range: tuple[float, float]
    duration_regime: str
    processing_time_bounds: str
    sample_count: int = 10


_TRACK3_SIZE_REGIMES: tuple[
    tuple[str, tuple[int, int], tuple[int, int], tuple[float, float]], ...
] = (
    ("S0_small_square", (10, 15), (10, 15), (0.8, 1.25)),
    ("S1_tall_rect", (20, 40), (10, 15), (1.5, 3.0)),
    ("S2_medium_square", (20, 30), (20, 30), (0.8, 1.25)),
    ("S3_large_tall", (40, 60), (15, 20), (2.0, 4.0)),
)

_TRACK3_DURATION_REGIMES: tuple[tuple[str, str], ...] = (
    ("D0_uniform_narrow", "Uniform discrete p in [1, 50]; low machine-load skew (CV < 0.10)"),
    ("D1_uniform_wide", "Uniform discrete p in [1, 100]; moderate load skew (0.10 <= CV < 0.25)"),
    ("D2_bottleneck_skew", "Mixed uniform with bottleneck machine load >= 1.35x mean machine load"),
    ("D3_bimodal_cluster", "Bimodal discrete p in [1, 20] U [80, 100]; clustered durations"),
)

TRACK3_STRATA: tuple[Track3StratumSpec, ...] = tuple(
    Track3StratumSpec(
        stratum_id=f"SM_STRATUM_{idx:02d}",
        size_regime=size_name,
        job_range=j_range,
        machine_range=m_range,
        aspect_ratio_range=ratio_range,
        duration_regime=dist_name,
        processing_time_bounds=dist_desc,
        sample_count=10,
    )
    for idx, ((size_name, j_range, m_range, ratio_range), (dist_name, dist_desc)) in enumerate(
        (s_spec, d_spec) for s_spec in _TRACK3_SIZE_REGIMES for d_spec in _TRACK3_DURATION_REGIMES
    )
)

TRACK3_ALL_IDS: tuple[str, ...] = tuple(f"sm{idx:03d}" for idx in range(1, 161))
TRACK3_DEVELOPMENT_IDS_ORDERED: tuple[str, ...] = tuple(f"sm{idx:03d}" for idx in range(1, 161, 4))
TRACK3_CONFIRMATORY_IDS_ORDERED: tuple[str, ...] = tuple(
    inst_id for inst_id in TRACK3_ALL_IDS if inst_id not in set(TRACK3_DEVELOPMENT_IDS_ORDERED)
)
TRACK3_DEVELOPMENT_IDS: frozenset[str] = frozenset(TRACK3_DEVELOPMENT_IDS_ORDERED)


def stratum_for_track3_instance(instance_id: str) -> Track3StratumSpec:
    """Return the pre-registered Track 3 feature stratum for `sm001`..`sm160`."""
    if not instance_id.startswith("sm") or not instance_id[2:].isdigit():
        raise KeyError(f"Invalid Track 3 instance ID: {instance_id!r}")
    numeric_idx = int(instance_id[2:])
    if numeric_idx < 1 or numeric_idx > 160:
        raise KeyError(f"Track 3 instance ID out of range 1..160: {instance_id!r}")
    stratum_idx = (numeric_idx - 1) // 10
    return TRACK3_STRATA[stratum_idx]


def split_for_track1_instance(instance_id: str) -> SplitAssignment:
    """Return frozen DEVELOPMENT or CONFIRMATORY split for a Track 1 instance."""
    if instance_id in TRACK1_DEVELOPMENT_IDS:
        return SplitAssignment.DEVELOPMENT
    return SplitAssignment.CONFIRMATORY


def split_for_track3_instance(instance_id: str) -> SplitAssignment:
    """Return frozen DEVELOPMENT (40) or CONFIRMATORY (120) split for Track 3."""
    if instance_id in TRACK3_DEVELOPMENT_IDS:
        return SplitAssignment.DEVELOPMENT
    return SplitAssignment.CONFIRMATORY
