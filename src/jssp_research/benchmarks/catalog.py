"""Reproducible benchmark catalog and integrity validation across Tracks 1–5."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from jssp_research.benchmarks.checksums import verify_sha256
from jssp_research.benchmarks.loaders import PARSER_REGISTRY, parse_jssp_file
from jssp_research.benchmarks.provenance import (
    AcquisitionStatus,
    BenchmarkCatalogEntry,
    BenchmarkTrack,
    BoundProvenance,
    BoundStatus,
    SourceProvenance,
    SplitAssignment,
)
from jssp_research.benchmarks.splits import (
    TRACK1_DEVELOPMENT_IDS,
    TRACK3_CONFIRMATORY_IDS_ORDERED,
    TRACK3_DATASET_ID,
    TRACK3_DATASET_VERSION,
    TRACK3_DEVELOPMENT_IDS,
    TRACK3_DEVELOPMENT_IDS_ORDERED,
    TRACK3_PRE_OUTCOME_BLINDED,
    TRACK3_SELECTION_METHOD,
    TRACK3_SELECTION_SEED,
    TRACK3_SOURCE_ARCHIVE_SHA256,
    TRACK3_SOURCE_VERIFICATION_STATUS,
    split_for_track1_instance,
    split_for_track3_instance,
    stratum_for_track3_instance,
)

DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[3]
EXPECTED_FIXTURES_REL_PATH = Path("tests/fixtures/expected_schedules.json")


@dataclass(frozen=True, slots=True)
class FamilySummaryRow:
    """Per-(track, family) catalog statistics."""

    track: BenchmarkTrack
    family: str
    total: int
    available_locally: int
    development: int
    confirmatory_or_held_out: int
    deferred: int


@dataclass(frozen=True, slots=True)
class CatalogValidationReport:
    """Result summary emitted by `validate_catalog`.

    Explicitly distinguishes catalog metadata integrity verification (`metadata_entries_validated`)
    from actual benchmark file parsing and checksum validation (`local_files_parsed_and_verified`).
    """

    total_entries: int
    metadata_entries_validated: int
    track_counts: dict[str, int]
    available_files_verified: int
    local_files_parsed_and_verified: int
    pending_file_acquisition_entries: int
    deferred_entries: int
    development_entries: int
    confirmatory_entries: int
    errors: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0


def _load_fixture_metadata(repo_root: Path) -> dict[str, dict[str, object]]:
    meta_path = repo_root / EXPECTED_FIXTURES_REL_PATH
    if not meta_path.is_file():
        return {}
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    fixtures = payload.get("fixtures", {})
    return fixtures if isinstance(fixtures, dict) else {}


def _track1_specs() -> list[tuple[str, str, int, int, BoundProvenance, SourceProvenance]]:
    jsplib_url = "https://github.com/tamy0612/JSPLib"
    orlib_terms = "Academic benchmark archive (JSPLib / OR-Library)"
    pending_bounds = BoundProvenance(
        status=BoundStatus.PENDING_SNAPSHOT,
        source="pending JSPLib/BKS snapshot acquisition",
        recorded_at="2026-11-23",
    )

    specs: list[tuple[str, str, int, int, BoundProvenance, SourceProvenance]] = []

    # FT (3)
    ft_source = SourceProvenance(
        citation="H. Fisher and G. L. Thompson (1963), Industrial Scheduling, pp. 225-251",
        source_url=jsplib_url,
        license_or_terms=orlib_terms,
    )
    ft_optima = {"ft06": (6, 6, 55), "ft10": (10, 10, 930), "ft20": (20, 5, 1165)}
    for inst_id, (j, m, opt) in ft_optima.items():
        specs.append(
            (
                inst_id,
                "FT",
                j,
                m,
                BoundProvenance(
                    status=BoundStatus.PROVEN_OPTIMAL,
                    known_optimum=opt,
                    best_known_upper_bound=opt,
                    lower_bound=opt,
                    source="Fisher & Thompson (1963); Carlier & Pinson (1989); JSPLib",
                    recorded_at="2026-11-23",
                ),
                ft_source,
            )
        )

    # LA (40)
    la_source = SourceProvenance(
        citation="S. Lawrence (1984), GSIA Carnegie Mellon University",
        source_url=jsplib_url,
        license_or_terms=orlib_terms,
    )
    la_sizes = [
        (1, 5, 10, 5),
        (6, 10, 15, 5),
        (11, 15, 20, 5),
        (16, 20, 10, 10),
        (21, 25, 15, 10),
        (26, 30, 20, 10),
        (31, 35, 30, 10),
        (36, 40, 15, 15),
    ]
    for start, end, j, m in la_sizes:
        for idx in range(start, end + 1):
            specs.append((f"la{idx:02d}", "LA", j, m, pending_bounds, la_source))

    # ABZ (5)
    abz_source = SourceProvenance(
        citation="J. Adams, E. Balas, and D. Zawack (1988), Management Science 34(3):391-401",
        source_url=jsplib_url,
        license_or_terms=orlib_terms,
    )
    for idx in range(5, 10):
        j, m = (10, 10) if idx <= 6 else (20, 15)
        specs.append((f"abz{idx}", "ABZ", j, m, pending_bounds, abz_source))

    # ORB (10)
    orb_source = SourceProvenance(
        citation="D. Applegate and W. Cook (1991), ORSA J. Computing 3(2):149-156",
        source_url=jsplib_url,
        license_or_terms=orlib_terms,
    )
    for idx in range(1, 11):
        specs.append((f"orb{idx:02d}", "ORB", 10, 10, pending_bounds, orb_source))

    # SWV (20)
    swv_source = SourceProvenance(
        citation=(
            "R. H. Storer, S. D. Wu, and R. Vaccari (1992), Management Science 38(10):1495-1509"
        ),
        source_url=jsplib_url,
        license_or_terms=orlib_terms,
    )
    for idx in range(1, 21):
        if idx <= 5:
            j, m = 20, 10
        elif idx <= 10:
            j, m = 20, 15
        else:
            j, m = 50, 10
        specs.append((f"swv{idx:02d}", "SWV", j, m, pending_bounds, swv_source))

    # YN (4)
    yn_source = SourceProvenance(
        citation="T. Yamada and R. Nakano (1992), PPSN II, pp. 281-290",
        source_url=jsplib_url,
        license_or_terms=orlib_terms,
    )
    for idx in range(1, 5):
        specs.append((f"yn{idx}", "YN", 20, 20, pending_bounds, yn_source))

    # TA (80)
    ta_source = SourceProvenance(
        citation="E. Taillard (1993), European J. Operational Research 64(2):278-285",
        source_url=jsplib_url,
        license_or_terms=orlib_terms,
    )
    ta_strata = [
        (1, 10, 15, 15),
        (11, 20, 20, 15),
        (21, 30, 20, 20),
        (31, 40, 30, 15),
        (41, 50, 30, 20),
        (51, 60, 50, 15),
        (61, 70, 50, 20),
        (71, 80, 100, 20),
    ]
    for start, end, j, m in ta_strata:
        for idx in range(start, end + 1):
            specs.append((f"ta{idx:02d}", "TA", j, m, pending_bounds, ta_source))

    # DMU (80)
    dmu_source = SourceProvenance(
        citation=(
            "E. Demirkol, S. Mehta, and R. Uzsoy (1998), European J. Operational Research "
            "109(1):137-148"
        ),
        source_url=jsplib_url,
        license_or_terms=orlib_terms,
    )
    dmu_strata = [
        (1, 10, 20, 15),
        (11, 20, 20, 20),
        (21, 30, 30, 15),
        (31, 40, 30, 20),
        (41, 50, 40, 15),
        (51, 60, 40, 20),
        (61, 70, 50, 15),
        (71, 80, 50, 20),
    ]
    for start, end, j, m in dmu_strata:
        for idx in range(start, end + 1):
            specs.append((f"dmu{idx:02d}", "DMU", j, m, pending_bounds, dmu_source))

    return specs


def build_benchmark_catalog(repo_root: Path = DEFAULT_REPO_ROOT) -> list[BenchmarkCatalogEntry]:
    """Construct the complete 536-entry benchmark catalog across Tracks 1–5."""
    fixture_meta = _load_fixture_metadata(repo_root)
    entries: list[BenchmarkCatalogEntry] = []

    for inst_id, family, j, m, bounds, src in _track1_specs():
        split = split_for_track1_instance(inst_id)
        local_meta = fixture_meta.get(inst_id)
        if local_meta is not None:
            rel_path = str(local_meta["relative_path"])
            status = AcquisitionStatus.COMMITTED_FIXTURE
            raw_sha = str(local_meta["raw_sha256"])
            canon_sha = str(local_meta["canonical_sha256"])
            src = SourceProvenance(
                citation=src.citation,
                source_url=src.source_url,
                retrieval_date=str(local_meta.get("retrieval_date", "2026-11-23")),
                license_or_terms=str(local_meta.get("license", src.license_or_terms)),
            )
        else:
            rel_path = None
            status = AcquisitionStatus.METADATA_CATALOGED_FILE_PENDING
            raw_sha = None
            canon_sha = None

        entries.append(
            BenchmarkCatalogEntry(
                instance_id=inst_id,
                family=family,
                track=BenchmarkTrack.TRACK1_CORE_CLASSICAL,
                status=status,
                split=split,
                num_jobs=j,
                num_machines=m,
                operation_count=j * m,
                processing_time_distribution="classical_integer_pending_file_scan"
                if rel_path is None
                else "classical_discrete_uniform",
                format_id="standard_jssp_0based_v1",
                parser_version="1.0.0",
                local_relative_path=rel_path,
                raw_sha256=raw_sha,
                canonical_sha256=canon_sha,
                source_provenance=src,
                bounds=bounds,
            )
        )

    pending_bounds = BoundProvenance(
        status=BoundStatus.PENDING_SNAPSHOT,
        source="pending dataset acquisition and bound snapshot",
        recorded_at="2026-11-23",
    )

    # Track 2a: 80 Large-TA rectangular instances (strata below 1000x1000)
    large_ta_source = SourceProvenance(
        citation="G. Da Col and E. Teppan (2019), Large-TA JSSP Benchmarks",
        source_url="https://github.com/teppan/large-jssp",
        license_or_terms="Academic benchmark archive (file acquisition pending)",
    )
    for idx in range(1, 81):
        entries.append(
            BenchmarkCatalogEntry(
                instance_id=f"lta{idx:02d}",
                family="LARGE_TA",
                track=BenchmarkTrack.TRACK2A_LARGE_TA,
                status=AcquisitionStatus.METADATA_CATALOGED_FILE_PENDING,
                split=SplitAssignment.SCALABILITY_HELD_OUT,
                num_jobs=None,
                num_machines=None,
                operation_count=None,
                processing_time_distribution="pending_source_file",
                format_id="standard_jssp_0based_v1",
                parser_version="1.0.0",
                source_provenance=large_ta_source,
                bounds=pending_bounds,
                notes="Large-TA rectangular stratum < 1000x1000; JxM populated on acquisition.",
            )
        )

    # Track 2b: 10 Extreme Stress instances (1000x1000)
    for idx in range(81, 91):
        entries.append(
            BenchmarkCatalogEntry(
                instance_id=f"lta{idx:02d}",
                family="STRESS_TAI",
                track=BenchmarkTrack.TRACK2B_STRESS,
                status=AcquisitionStatus.METADATA_CATALOGED_FILE_PENDING,
                split=SplitAssignment.STRESS_HELD_OUT,
                num_jobs=1000,
                num_machines=1000,
                operation_count=1_000_000,
                processing_time_distribution="pending_source_file",
                format_id="standard_jssp_0based_v1",
                parser_version="1.0.0",
                source_provenance=large_ta_source,
                bounds=pending_bounds,
                notes="Extreme 1000x1000 stress stratum (10^6 ops; excluded from standard Sobol).",
            )
        )

    # Track 3: 160 Stratified Strassl-Musliu synthetic instances (40 dev / 120 confirmatory)
    sm_source = SourceProvenance(
        citation="S. Strassl and N. Musliu (2022), Computers & Operations Research 141:105661",
        source_url="https://doi.org/10.1016/j.cor.2021.105661",
        license_or_terms="Academic synthetic dataset (source archive pending acquisition)",
    )
    for idx in range(1, 161):
        inst_id = f"sm{idx:03d}"
        stratum = stratum_for_track3_instance(inst_id)
        entries.append(
            BenchmarkCatalogEntry(
                instance_id=inst_id,
                family="STRASSL_MUSLIU",
                track=BenchmarkTrack.TRACK3_STRASSL_MUSLIU,
                status=AcquisitionStatus.METADATA_CATALOGED_FILE_PENDING,
                split=split_for_track3_instance(inst_id),
                num_jobs=None,
                num_machines=None,
                operation_count=None,
                processing_time_distribution=(
                    f"{stratum.duration_regime} ({stratum.processing_time_bounds})"
                ),
                format_id="standard_jssp_0based_v1",
                parser_version="1.0.0",
                source_provenance=sm_source,
                bounds=pending_bounds,
                notes=(
                    f"dataset={TRACK3_DATASET_ID} ({TRACK3_DATASET_VERSION}); "
                    f"stratum={stratum.stratum_id} [{stratum.size_regime}, "
                    f"J={stratum.job_range}, M={stratum.machine_range}, "
                    f"J/M={stratum.aspect_ratio_range}]; "
                    f"sampling={TRACK3_SELECTION_METHOD}; seed={TRACK3_SELECTION_SEED}; "
                    f"pre_outcome_blinded={TRACK3_PRE_OUTCOME_BLINDED}; "
                    f"source_status={TRACK3_SOURCE_VERIFICATION_STATUS}; "
                    f"archive_sha256={TRACK3_SOURCE_ARCHIVE_SHA256}"
                ),
            )
        )

    # Track 4: 24 Variable-Route Known-Optima instances (Explicitly Deferred per D07-C)
    track4_source = SourceProvenance(
        citation="G. Da Col and E. Teppan (2019), Variable-Route Known-Optima JSSP Set",
        source_url="https://github.com/teppan/large-jssp",
        license_or_terms="Academic benchmark archive (deferred extension track)",
    )
    for idx in range(1, 25):
        entries.append(
            BenchmarkCatalogEntry(
                instance_id=f"dct_ko{idx:02d}",
                family="DCT_VARIABLE_ROUTE",
                track=BenchmarkTrack.TRACK4_VARIABLE_ROUTE,
                status=AcquisitionStatus.DEFERRED_UNSUPPORTED_MODEL,
                split=SplitAssignment.DEFERRED,
                processing_time_distribution="deferred_variable_route",
                source_provenance=track4_source,
                bounds=BoundProvenance(
                    status=BoundStatus.PENDING_SNAPSHOT,
                    source="deferred pending variable-route OBR/graph support (D07-C)",
                    recorded_at="2026-11-23",
                ),
                notes="Deferred per D07-C: variable routes / machine revisits violate OBR.",
            )
        )

    # Track 5: 20 BEL Industrial instances (Explicitly Deferred per D07-C)
    track5_source = SourceProvenance(
        citation="BEL Industrial Scheduling Instances (see RESEARCH_PLAN_V1.1.md Track 5)",
        source_url="pending_verified_industrial_archive",
        license_or_terms="Industrial benchmark terms pending format & Cmax verification",
    )
    for idx in range(1, 21):
        entries.append(
            BenchmarkCatalogEntry(
                instance_id=f"bel{idx:02d}",
                family="BEL_INDUSTRIAL",
                track=BenchmarkTrack.TRACK5_BEL_INDUSTRIAL,
                status=AcquisitionStatus.DEFERRED_UNSUPPORTED_MODEL,
                split=SplitAssignment.DEFERRED,
                processing_time_distribution="deferred_industrial",
                source_provenance=track5_source,
                bounds=BoundProvenance(
                    status=BoundStatus.PENDING_SNAPSHOT,
                    source="deferred pending format & Cmax recomputation contract (D07-C)",
                    recorded_at="2026-11-23",
                ),
                notes="Deferred per D07-C: do not compare against weighted completion times.",
            )
        )

    return entries


def get_catalog_entry(
    instance_id: str,
    repo_root: Path = DEFAULT_REPO_ROOT,
) -> BenchmarkCatalogEntry:
    """Look up a single catalog entry by unique `instance_id`."""
    for entry in build_benchmark_catalog(repo_root=repo_root):
        if entry.instance_id == instance_id:
            return entry
    raise KeyError(f"Unknown benchmark instance ID: {instance_id!r}")


def list_catalog_entries(
    *,
    track: BenchmarkTrack | None = None,
    family: str | None = None,
    split: SplitAssignment | None = None,
    repo_root: Path = DEFAULT_REPO_ROOT,
) -> list[BenchmarkCatalogEntry]:
    """Filter catalog entries by optional track, family (case-insensitive), and split."""
    entries = build_benchmark_catalog(repo_root=repo_root)
    if track is not None:
        entries = [e for e in entries if e.track == track]
    if family is not None:
        family_upper = family.upper()
        entries = [e for e in entries if e.family.upper() == family_upper]
    if split is not None:
        entries = [e for e in entries if e.split == split]
    return entries


def family_summary(repo_root: Path = DEFAULT_REPO_ROOT) -> list[FamilySummaryRow]:
    """Summarize catalog entry counts grouped in deterministic order by `(track, family)`."""
    entries = build_benchmark_catalog(repo_root=repo_root)
    buckets: dict[tuple[BenchmarkTrack, str], list[BenchmarkCatalogEntry]] = {}
    for entry in entries:
        key = (entry.track, entry.family)
        buckets.setdefault(key, []).append(entry)

    rows: list[FamilySummaryRow] = []
    for (track, family), group in buckets.items():
        rows.append(
            FamilySummaryRow(
                track=track,
                family=family,
                total=len(group),
                available_locally=sum(
                    1 for e in group if e.status == AcquisitionStatus.COMMITTED_FIXTURE
                ),
                development=sum(1 for e in group if e.split == SplitAssignment.DEVELOPMENT),
                confirmatory_or_held_out=sum(
                    1
                    for e in group
                    if e.split
                    in {
                        SplitAssignment.CONFIRMATORY,
                        SplitAssignment.SCALABILITY_HELD_OUT,
                        SplitAssignment.STRESS_HELD_OUT,
                    }
                ),
                deferred=sum(1 for e in group if e.split == SplitAssignment.DEFERRED),
            )
        )
    return rows


def validate_catalog(repo_root: Path = DEFAULT_REPO_ROOT) -> CatalogValidationReport:
    """Validate all catalog uniqueness, parser, split, deferral, and fixture checksum invariants."""
    entries = build_benchmark_catalog(repo_root=repo_root)
    errors: list[str] = []

    seen_ids: set[str] = set()
    track_counts: dict[str, int] = {t.value: 0 for t in BenchmarkTrack}
    available_files_verified = 0
    deferred_entries = 0
    development_entries = 0
    confirmatory_entries = 0

    for entry in entries:
        if entry.instance_id in seen_ids:
            errors.append(f"Duplicate catalog ID: {entry.instance_id}")
        seen_ids.add(entry.instance_id)

        track_counts[entry.track.value] += 1
        if entry.split == SplitAssignment.DEFERRED:
            deferred_entries += 1
        elif entry.split == SplitAssignment.DEVELOPMENT:
            development_entries += 1
        elif entry.split == SplitAssignment.CONFIRMATORY:
            confirmatory_entries += 1

        if entry.format_id is not None and entry.format_id not in PARSER_REGISTRY:
            errors.append(
                f"{entry.instance_id}: referenced parser {entry.format_id!r} not in PARSER_REGISTRY"
            )

        if entry.track in {
            BenchmarkTrack.TRACK4_VARIABLE_ROUTE,
            BenchmarkTrack.TRACK5_BEL_INDUSTRIAL,
        }:
            if entry.status != AcquisitionStatus.DEFERRED_UNSUPPORTED_MODEL:
                errors.append(f"{entry.instance_id}: unsupported track not marked deferred")
            if entry.format_id is not None:
                errors.append(
                    f"{entry.instance_id}: unsupported track unexpectedly references parser"
                )

        if entry.status == AcquisitionStatus.COMMITTED_FIXTURE:
            if not entry.local_relative_path:
                errors.append(f"{entry.instance_id}: COMMITTED_FIXTURE missing local_relative_path")
                continue
            file_path = repo_root / entry.local_relative_path
            if not file_path.is_file():
                errors.append(f"{entry.instance_id}: missing local file {file_path}")
                continue
            if entry.format_id is None:
                errors.append(f"{entry.instance_id}: missing format_id for committed fixture")
                continue
            try:
                parsed = parse_jssp_file(
                    file_path,
                    instance_id=entry.instance_id,
                    format_id=entry.format_id,
                )
                if entry.num_jobs is not None and parsed.num_jobs != entry.num_jobs:
                    errors.append(
                        f"{entry.instance_id}: num_jobs {parsed.num_jobs} "
                        f"!= catalog {entry.num_jobs}"
                    )
                if entry.num_machines is not None and parsed.num_machines != entry.num_machines:
                    errors.append(
                        f"{entry.instance_id}: num_machines {parsed.num_machines} "
                        f"!= catalog {entry.num_machines}"
                    )
                if entry.raw_sha256 is not None:
                    verify_sha256(
                        parsed.raw_sha256,
                        entry.raw_sha256,
                        context=f"{entry.instance_id} raw_sha256",
                    )
                if entry.canonical_sha256 is not None:
                    verify_sha256(
                        parsed.canonical_sha256,
                        entry.canonical_sha256,
                        context=f"{entry.instance_id} canonical_sha256",
                    )
                available_files_verified += 1
            except Exception as exc:
                errors.append(f"{entry.instance_id}: fixture validation failed: {exc}")

    # Verify expected split and track counts
    expected_tracks = {
        BenchmarkTrack.TRACK1_CORE_CLASSICAL.value: 242,
        BenchmarkTrack.TRACK2A_LARGE_TA.value: 80,
        BenchmarkTrack.TRACK2B_STRESS.value: 10,
        BenchmarkTrack.TRACK3_STRASSL_MUSLIU.value: 160,
        BenchmarkTrack.TRACK4_VARIABLE_ROUTE.value: 24,
        BenchmarkTrack.TRACK5_BEL_INDUSTRIAL.value: 20,
    }
    for key, exp_count in expected_tracks.items():
        if track_counts.get(key) != exp_count:
            errors.append(f"Track {key} count {track_counts.get(key)} != expected {exp_count}")

    if len(TRACK1_DEVELOPMENT_IDS) != 48:
        errors.append(
            f"TRACK1_DEVELOPMENT_IDS has {len(TRACK1_DEVELOPMENT_IDS)} entries, expected 48"
        )
    if len(TRACK3_DEVELOPMENT_IDS) != 40:
        errors.append(
            f"TRACK3_DEVELOPMENT_IDS has {len(TRACK3_DEVELOPMENT_IDS)} entries, expected 40"
        )
    if len(TRACK3_DEVELOPMENT_IDS_ORDERED) != 40 or len(TRACK3_CONFIRMATORY_IDS_ORDERED) != 120:
        errors.append("Track 3 dev/confirmatory ordered split counts must equal 40 and 120")

    pending_file_entries = sum(
        1 for e in entries if e.status == AcquisitionStatus.METADATA_CATALOGED_FILE_PENDING
    )

    return CatalogValidationReport(
        total_entries=len(entries),
        metadata_entries_validated=len(entries),
        track_counts=track_counts,
        available_files_verified=available_files_verified,
        local_files_parsed_and_verified=available_files_verified,
        pending_file_acquisition_entries=pending_file_entries,
        deferred_entries=deferred_entries,
        development_entries=development_entries,
        confirmatory_entries=confirmatory_entries,
        errors=tuple(errors),
    )
