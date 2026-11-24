"""Python tests for Rust RNG seed derivation, pairing policy, benchmark parsers, and catalog."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from jssp_research.benchmarks import (
    TRACK3_ALL_IDS,
    TRACK3_CONFIRMATORY_IDS_ORDERED,
    TRACK3_DATASET_ID,
    TRACK3_DEVELOPMENT_IDS_ORDERED,
    TRACK3_PRE_OUTCOME_BLINDED,
    TRACK3_SELECTION_METHOD,
    TRACK3_SELECTION_SEED,
    TRACK3_SOURCE_ARCHIVE_SHA256,
    TRACK3_SOURCE_VERIFICATION_STATUS,
    TRACK3_STRATA,
    AcquisitionStatus,
    BenchmarkParseError,
    BenchmarkTrack,
    BoundProvenance,
    BoundStatus,
    ChecksumMismatchError,
    SplitAssignment,
    UnsupportedBenchmarkFormatError,
    build_benchmark_catalog,
    parse_jssp_file,
    parse_jssp_text,
    stratum_for_track3_instance,
    validate_catalog,
    verify_sha256,
)
from jssp_research.experiments import (
    PairingMode,
    build_seed_context,
    frozen_master_seed_for_replicate,
    resolve_named_streams,
    rust_rng_metadata,
)
from jssp_research.native import rng_sample_range_and_shuffle

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"


def test_frozen_master_seed_schedule_and_rust_rng_metadata() -> None:
    """Verify exact Rust ChaCha20 RNG metadata and SHA-256 frozen master seed schedule."""
    meta = rust_rng_metadata()
    assert meta.algorithm == "ChaCha20Rng"
    assert meta.rng_crate_version == "rand_chacha-0.10.0"
    assert meta.hash_crate_version == "sha2-0.11.1"
    assert meta.chacha_rounds == 20
    assert len(meta.stream_roles) == 9

    for r in range(20):
        expected = int.from_bytes(
            hashlib.sha256(f"jssp-research-v1-seed-{r}".encode("ascii")).digest()[:8],
            "little",
        )
        assert frozen_master_seed_for_replicate(r) == expected

    with pytest.raises(ValueError, match="outside frozen schedule"):
        frozen_master_seed_for_replicate(20)


def test_python_rust_stream_derivation_pairing_and_primitives() -> None:
    """Verify paired vs configuration-scoped derivation and reference primitive outputs."""
    ctx_b01 = build_seed_context(
        replicate_index=0,
        instance_id="ft06",
        pairing_mode=PairingMode.PAIRED_INIT,
        pairing_group="phase_b_shared",
        configuration_id="B01",
    )
    ctx_b02 = build_seed_context(
        replicate_index=0,
        instance_id="ft06",
        pairing_mode=PairingMode.PAIRED_INIT,
        pairing_group="phase_b_shared",
        configuration_id="B02",
    )

    streams_b01 = {s.role: s for s in resolve_named_streams(ctx_b01, preview_count=4)}
    streams_b02 = {s.role: s for s in resolve_named_streams(ctx_b02, preview_count=4)}

    assert streams_b01["init"] == streams_b02["init"]
    assert streams_b01["qmc_scramble"] == streams_b02["qmc_scramble"]
    assert streams_b01["perturbation"].seed_hex != streams_b02["perturbation"].seed_hex
    assert (
        streams_b01["local_search_sample"].seed_hex != streams_b02["local_search_sample"].seed_hex
    )

    # Reference vector cross-check across PyO3
    ref_ctx = build_seed_context(
        replicate_index=3,
        instance_id="abz5",
        pairing_mode=PairingMode.CONFIG_SCOPED,
        configuration_id="M3_REF_V1",
        explicit_master_seed=0x1234_5678_9ABC_DEF0,
    )
    ref_init = {s.role: s for s in resolve_named_streams(ref_ctx, preview_count=4)}["init"]
    assert ref_init.seed_hex == "ea9abd3c2ced43d77e88dea00ffeee1caf3e4102a922fced131a7aaa07798936"
    samples, perm = rng_sample_range_and_shuffle(ref_init.seed_hex, 10, 8, 10)
    assert samples == [9, 6, 7, 0, 0, 7, 1, 1]
    assert perm == [9, 5, 4, 8, 1, 6, 3, 7, 2, 0]


def test_benchmark_parsers_fixtures_and_error_cases() -> None:
    """Verify valid multi-format parsing, checksum verification, and strict error handling."""
    fixture_doc = json.loads((FIXTURES_DIR / "expected_schedules.json").read_text("utf-8"))

    tiny_meta = fixture_doc["fixtures"]["tiny_jssp"]
    tiny = parse_jssp_file(REPO_ROOT / tiny_meta["relative_path"], instance_id="tiny_jssp")
    assert (tiny.num_jobs, tiny.num_machines) == (3, 3)
    verify_sha256(tiny.raw_sha256, tiny_meta["raw_sha256"])
    verify_sha256(tiny.canonical_sha256, tiny_meta["canonical_sha256"])

    ft06_meta = fixture_doc["fixtures"]["ft06"]
    ft06 = parse_jssp_file(REPO_ROOT / ft06_meta["relative_path"], instance_id="ft06")
    assert (ft06.num_jobs, ft06.num_machines) == (6, 6)
    verify_sha256(ft06.raw_sha256, ft06_meta["raw_sha256"])
    verify_sha256(ft06.canonical_sha256, ft06_meta["canonical_sha256"])

    emb_1b = fixture_doc["embedded_format_fixtures"]["tiny_jssp_1based"]
    parsed_1b = parse_jssp_text(
        emb_1b["text"],
        instance_id="tiny_jssp_1based",
        format_id=emb_1b["format_id"],
    )
    verify_sha256(parsed_1b.canonical_sha256, emb_1b["canonical_sha256"])

    emb_tai = fixture_doc["embedded_format_fixtures"]["ft06_taillard"]
    parsed_tai = parse_jssp_text(
        emb_tai["text"],
        instance_id="ft06_taillard",
        format_id=emb_tai["format_id"],
    )
    verify_sha256(parsed_tai.canonical_sha256, emb_tai["canonical_sha256"])
    assert parsed_tai.canonical_sha256 == ft06.canonical_sha256

    # Error cases
    with pytest.raises(UnsupportedBenchmarkFormatError):
        parse_jssp_text("1 1\n0 1\n", instance_id="x", format_id="unknown_fmt")

    with pytest.raises(BenchmarkParseError, match="expected 2 job rows"):
        parse_jssp_text("2 2\n0 1 1 2\n", instance_id="short")

    with pytest.raises(BenchmarkParseError, match="must contain 4 integers"):
        parse_jssp_text("2 2\n0 1 1 2\n0 1\n", instance_id="nonrect")

    with pytest.raises(BenchmarkParseError, match="non-positive processing time"):
        parse_jssp_text("2 2\n0 0 1 0\n0 1 1 1\n", instance_id="zero_job_work")

    # Individual zero-duration operation inside a positive-work job (e.g. orb07) is valid.
    parsed_zero_op = parse_jssp_text("2 2\n0 1 1 0\n0 1 1 1\n", instance_id="zero_op_ok")
    assert parsed_zero_op.total_processing_time == 3

    with pytest.raises(BenchmarkParseError, match="references machine 2"):
        parse_jssp_text("2 2\n0 1 2 1\n0 1 1 1\n", instance_id="bad_mach")

    with pytest.raises(BenchmarkParseError, match="visits machine 0 more than once"):
        parse_jssp_text("2 2\n0 1 0 1\n0 1 1 1\n", instance_id="dup_mach")

    with pytest.raises(ChecksumMismatchError, match="SHA-256 mismatch"):
        verify_sha256("0" * 64, tiny.raw_sha256, context="tiny_jssp")

    with pytest.raises(ValueError, match="known_optimum must only be set"):
        BoundProvenance(
            status=BoundStatus.BEST_KNOWN_OPEN,
            known_optimum=55,
            best_known_upper_bound=55,
            source="test",
            recorded_at="2026-11-23",
        )


def test_complete_catalog_counts_splits_and_deferrals() -> None:
    """Verify full 536-instance catalog invariants, splits, and Track 4/5 deferrals."""
    entries = build_benchmark_catalog(repo_root=REPO_ROOT)
    assert len(entries) == 536

    report = validate_catalog(repo_root=REPO_ROOT)
    assert report.is_valid, f"Catalog validation failed: {report.errors}"
    assert report.track_counts == {
        BenchmarkTrack.TRACK1_CORE_CLASSICAL.value: 242,
        BenchmarkTrack.TRACK2A_LARGE_TA.value: 80,
        BenchmarkTrack.TRACK2B_STRESS.value: 10,
        BenchmarkTrack.TRACK3_STRASSL_MUSLIU.value: 160,
        BenchmarkTrack.TRACK4_VARIABLE_ROUTE.value: 24,
        BenchmarkTrack.TRACK5_BEL_INDUSTRIAL.value: 20,
    }
    track1_raw_dir = REPO_ROOT / "data" / "raw" / "track1"
    if (track1_raw_dir / "ft" / "ft06.txt").is_file():
        assert report.available_files_verified == 242
        assert report.local_files_parsed_and_verified == 242
        assert report.pending_file_acquisition_entries == 250
    else:
        assert report.available_files_verified == 1
        assert report.local_files_parsed_and_verified == 1
        assert report.pending_file_acquisition_entries == 491
    assert report.metadata_entries_validated == 536
    assert report.deferred_entries == 44

    t1_dev = [
        e
        for e in entries
        if e.track == BenchmarkTrack.TRACK1_CORE_CLASSICAL
        and e.split == SplitAssignment.DEVELOPMENT
    ]
    t1_conf = [
        e
        for e in entries
        if e.track == BenchmarkTrack.TRACK1_CORE_CLASSICAL
        and e.split == SplitAssignment.CONFIRMATORY
    ]
    assert len(t1_dev) == 48
    assert len(t1_conf) == 194

    t3_dev = [
        e
        for e in entries
        if e.track == BenchmarkTrack.TRACK3_STRASSL_MUSLIU
        and e.split == SplitAssignment.DEVELOPMENT
    ]
    t3_conf = [
        e
        for e in entries
        if e.track == BenchmarkTrack.TRACK3_STRASSL_MUSLIU
        and e.split == SplitAssignment.CONFIRMATORY
    ]
    assert len(t3_dev) == 40
    assert len(t3_conf) == 120

    deferred = [
        e
        for e in entries
        if e.track in {BenchmarkTrack.TRACK4_VARIABLE_ROUTE, BenchmarkTrack.TRACK5_BEL_INDUSTRIAL}
    ]
    assert all(e.status == AcquisitionStatus.DEFERRED_UNSUPPORTED_MODEL for e in deferred)
    assert all(e.format_id is None for e in deferred)


def test_e01_shared_stream_seeds_vs_crn_alias() -> None:
    """Verify `shared_stream_seeds` canonical naming, `crn` alias, and pairing bounds."""
    ctx_shared = build_seed_context(
        replicate_index=0,
        instance_id="ft06",
        pairing_mode=PairingMode.SHARED_STREAM_SEEDS,
        pairing_group="phase_b_shared",
    )
    ctx_alias = build_seed_context(
        replicate_index=0,
        instance_id="ft06",
        pairing_mode="crn",
        pairing_group="phase_b_shared",
    )
    assert ctx_shared.pairing_mode == PairingMode.SHARED_STREAM_SEEDS
    assert ctx_alias == ctx_shared
    assert resolve_named_streams(ctx_shared, preview_count=4) == resolve_named_streams(
        ctx_alias, preview_count=4
    )


def test_e04_track3_strassl_musliu_sample_and_split_provenance() -> None:
    """Verify explicit Track 3 16-stratum sample provenance and blinded 40/120 split."""
    assert TRACK3_DATASET_ID == "strassl_musliu_2022_jssp_4225"
    assert TRACK3_SOURCE_ARCHIVE_SHA256 is None
    assert TRACK3_SOURCE_VERIFICATION_STATUS == "pending_raw_archive_acquisition"
    assert TRACK3_SELECTION_METHOD == "stratified_16_cell_equal_allocation_10_per_stratum"
    assert "0x5433534D32303236" in TRACK3_SELECTION_SEED
    assert TRACK3_PRE_OUTCOME_BLINDED is True

    assert len(TRACK3_STRATA) == 16
    assert sum(s.sample_count for s in TRACK3_STRATA) == 160
    assert len(TRACK3_ALL_IDS) == 160
    assert len(TRACK3_DEVELOPMENT_IDS_ORDERED) == 40
    assert len(TRACK3_CONFIRMATORY_IDS_ORDERED) == 120
    assert set(TRACK3_DEVELOPMENT_IDS_ORDERED).isdisjoint(TRACK3_CONFIRMATORY_IDS_ORDERED)
    assert set(TRACK3_DEVELOPMENT_IDS_ORDERED) | set(TRACK3_CONFIRMATORY_IDS_ORDERED) == set(
        TRACK3_ALL_IDS
    )

    stratum_counts: dict[str, int] = {}
    for inst_id in TRACK3_ALL_IDS:
        stratum = stratum_for_track3_instance(inst_id)
        stratum_counts[stratum.stratum_id] = stratum_counts.get(stratum.stratum_id, 0) + 1
    assert len(stratum_counts) == 16
    assert all(cnt == 10 for cnt in stratum_counts.values())

    sm001_entry = next(
        e for e in build_benchmark_catalog(repo_root=REPO_ROOT) if e.instance_id == "sm001"
    )
    assert sm001_entry.raw_sha256 is None
    assert sm001_entry.canonical_sha256 is None
    assert "SM_STRATUM_00" in sm001_entry.notes
    assert "pending_raw_archive_acquisition" in sm001_entry.notes
