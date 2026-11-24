"""Verify local Track 1 benchmark files, SHA-256 digests, bounds, and M3 splits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jssp_research.benchmarks import (
    AcquisitionStatus,
    BenchmarkTrack,
    SplitAssignment,
    parse_jssp_file,
    validate_catalog,
    verify_sha256,
)
from jssp_research.benchmarks.splits import TRACK1_DEVELOPMENT_IDS

REPO_ROOT = Path(__file__).resolve().parents[1]
METADATA_PATH = REPO_ROOT / "data" / "metadata" / "track1_checksums_and_provenance.json"
BOUNDS_PATH = REPO_ROOT / "data" / "bounds" / "track1_bks_snapshot_20261124.json"
SPLIT_PATH = REPO_ROOT / "data" / "splits" / "track1_split_v1.json"


def verify_all_track1_files(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Verify all 242 local Track 1 files, checksums, Rust validity, bounds, and split."""
    meta_doc = json.loads(
        (repo_root / "data/metadata/track1_checksums_and_provenance.json").read_text("utf-8")
    )
    bounds_doc = json.loads(
        (repo_root / "data/bounds/track1_bks_snapshot_20261124.json").read_text("utf-8")
    )
    split_doc = json.loads((repo_root / "data/splits/track1_split_v1.json").read_text("utf-8"))

    instances = meta_doc["instances"]
    if len(instances) != 242:
        raise ValueError(f"Expected 242 Track 1 metadata entries, found {len(instances)}")
    if set(split_doc["development_ids"]) != TRACK1_DEVELOPMENT_IDS:
        raise ValueError(
            "Track 1 development split JSON does not match frozen TRACK1_DEVELOPMENT_IDS"
        )
    if len(split_doc["confirmatory_ids"]) != 194:
        raise ValueError("Track 1 confirmatory split must contain 194 instances")

    bounds_by_id = {row["instance_id"]: row for row in bounds_doc["bounds"]}
    if len(bounds_by_id) != 242:
        raise ValueError(f"Expected 242 bounds entries, found {len(bounds_by_id)}")

    verified = 0
    for row in instances:
        inst_id = row["instance_id"]
        file_path = repo_root / row["raw_relative_path"]
        if not file_path.is_file():
            raise FileNotFoundError(f"Missing local Track 1 benchmark file: {file_path}")
        parsed = parse_jssp_file(file_path, instance_id=inst_id, format_id=row["format_id"])
        verify_sha256(parsed.raw_sha256, row["raw_sha256"], context=f"{inst_id} raw_sha256")
        verify_sha256(
            parsed.canonical_sha256, row["canonical_sha256"], context=f"{inst_id} canonical_sha256"
        )
        if parsed.num_jobs != row["num_jobs"] or parsed.num_machines != row["num_machines"]:
            raise ValueError(f"{inst_id}: parsed dimensions mismatch")
        if inst_id not in bounds_by_id:
            raise ValueError(f"{inst_id}: missing bounds entry")
        verified += 1

    catalog_report = validate_catalog(repo_root=repo_root)
    if not catalog_report.is_valid:
        raise ValueError(f"Catalog validation failed: {catalog_report.errors}")
    if catalog_report.local_files_parsed_and_verified != 242:
        raise ValueError(
            "Expected 242 verified local files in catalog, "
            f"got {catalog_report.local_files_parsed_and_verified}"
        )

    from jssp_research.benchmarks import list_catalog_entries

    t1_entries = list_catalog_entries(
        track=BenchmarkTrack.TRACK1_CORE_CLASSICAL, repo_root=repo_root
    )
    dev_count = sum(1 for e in t1_entries if e.split == SplitAssignment.DEVELOPMENT)
    conf_count = sum(1 for e in t1_entries if e.split == SplitAssignment.CONFIRMATORY)
    acquired_count = sum(
        1
        for e in t1_entries
        if e.status
        in {AcquisitionStatus.ACQUIRED_VERIFIED_LOCAL, AcquisitionStatus.COMMITTED_FIXTURE}
    )

    return {
        "verified_files": verified,
        "catalog_verified_files": catalog_report.local_files_parsed_and_verified,
        "track1_acquired": acquired_count,
        "track1_development": dev_count,
        "track1_confirmatory": conf_count,
        "proven_optimal": bounds_doc["proven_optimal_count"],
        "best_known_open": bounds_doc["best_known_open_count"],
    }


def main() -> None:
    report = verify_all_track1_files()
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
