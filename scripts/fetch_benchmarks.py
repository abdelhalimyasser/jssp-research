"""Acquire, parse, Rust-validate, and checksum all 242 Track 1 core JSSP benchmarks.

Downloads the 242 classical rectangular JSSP instances (`ft`, `la`, `abz`, `orb`,
`swv`, `yn`, `ta`, `dmu`) from `SchedulingLab/jsp-instances`, cross-verifies shared
metadata against `tamy0612/JSPLib` and the committed `ft06` fixture, parses every file
using M3's `parse_jssp_file` (backed by Rust `validate_instance`), and persists:
- `data/metadata/track1_checksums_and_provenance.json`
- `data/bounds/track1_bks_snapshot_20261124.json`
- `data/splits/track1_split_v1.json`
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jssp_research.benchmarks.loaders import parse_jssp_file
from jssp_research.benchmarks.splits import TRACK1_DEVELOPMENT_IDS

REPO_ROOT = Path(__file__).resolve().parents[1]

SCHEDULINGLAB_BASE_URL = "https://raw.githubusercontent.com/SchedulingLab/jsp-instances/main"
SCHEDULINGLAB_CLASSICAL_JSON_URL = f"{SCHEDULINGLAB_BASE_URL}/classical.json"
SCHEDULINGLAB_LICENSE_URL = f"{SCHEDULINGLAB_BASE_URL}/LICENSE"
JSPLIB_INSTANCES_JSON_URL = (
    "https://raw.githubusercontent.com/tamy0612/JSPLib/master/instances.json"
)

SNAPSHOT_DATE = "2026-11-24"
BOUNDS_SNAPSHOT_REL_PATH = Path(
    f"data/bounds/track1_bks_snapshot_{SNAPSHOT_DATE.replace('-', '')}.json"
)
METADATA_REL_PATH = Path("data/metadata/track1_checksums_and_provenance.json")
SPLIT_REL_PATH = Path("data/splits/track1_split_v1.json")

EXPECTED_FAMILY_COUNTS = {
    "ft": 3,
    "la": 40,
    "abz": 5,
    "orb": 10,
    "swv": 20,
    "yn": 4,
    "ta": 80,
    "dmu": 80,
}


def _fetch_bytes(url: str, timeout: float = 30.0) -> bytes:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "jssp-research-benchmark-fetcher/1.0"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _download_instance(
    item: dict[str, Any],
    raw_root: Path,
    *,
    force: bool = False,
) -> tuple[str, Path, bytes]:
    name = str(item["name"])
    rel_path = str(item["path"])
    url = f"{SCHEDULINGLAB_BASE_URL}/{rel_path}"
    dest = raw_root / rel_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and not force:
        return name, dest, dest.read_bytes()
    payload = _fetch_bytes(url)
    dest.write_bytes(payload)
    return name, dest, payload


def acquire_track1_benchmarks(
    repo_root: Path = REPO_ROOT,
    *,
    force: bool = False,
    workers: int = 8,
) -> dict[str, Any]:
    """Download, verify, parse, and snapshot all 242 Track 1 core JSSP instances."""
    raw_root = repo_root / "data" / "raw" / "track1"
    raw_root.mkdir(parents=True, exist_ok=True)

    classical_meta = json.loads(_fetch_bytes(SCHEDULINGLAB_CLASSICAL_JSON_URL).decode("utf-8"))
    jsplib_meta = json.loads(_fetch_bytes(JSPLIB_INSTANCES_JSON_URL).decode("utf-8"))
    jsplib_by_name: dict[str, dict[str, Any]] = {str(r["name"]): r for r in jsplib_meta}

    if len(classical_meta) != 242:
        raise RuntimeError(f"Expected 242 entries in classical.json, got {len(classical_meta)}")

    family_counts: dict[str, int] = {}
    for row in classical_meta:
        name = str(row["name"])
        fam = "".join(c for c in name if c.isalpha())
        family_counts[fam] = family_counts.get(fam, 0) + 1
    if family_counts != EXPECTED_FAMILY_COUNTS:
        raise RuntimeError(f"Family counts {family_counts} != expected {EXPECTED_FAMILY_COUNTS}")

    downloaded: dict[str, Path] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_download_instance, item, raw_root, force=force) for item in classical_meta
        ]
        for fut in concurrent.futures.as_completed(futures):
            name, dest, _ = fut.result()
            downloaded[name] = dest

    retrieval_utc = datetime.now(tz=UTC).isoformat()
    ft06_fixture = parse_jssp_file(
        repo_root / "tests" / "fixtures" / "medium_jssp.txt",
        instance_id="ft06",
        format_id="standard_jssp_0based_v1",
    )

    checksum_records: list[dict[str, Any]] = []
    bound_records: list[dict[str, Any]] = []
    dev_ids: list[str] = []
    confirmatory_ids: list[str] = []
    proven_count = 0
    open_count = 0
    cross_checked_jsplib = 0

    for item in sorted(classical_meta, key=lambda r: str(r["name"])):
        inst_id = str(item["name"])
        family = "".join(c for c in inst_id if c.isalpha()).upper()
        expected_j = int(item["jobs"])
        expected_m = int(item["machines"])
        raw_rel_from_repo = (Path("data/raw/track1") / str(item["path"])).as_posix()
        file_path = downloaded[inst_id]

        parsed = parse_jssp_file(
            file_path,
            instance_id=inst_id,
            format_id="standard_jssp_0based_v1",
        )
        if parsed.num_jobs != expected_j or parsed.num_machines != expected_m:
            raise ValueError(
                f"{inst_id}: parsed {parsed.num_jobs}x{parsed.num_machines} != "
                f"source {expected_j}x{expected_m}"
            )
        if inst_id == "ft06" and parsed.canonical_sha256 != ft06_fixture.canonical_sha256:
            raise ValueError(
                f"ft06 canonical digest {parsed.canonical_sha256} != committed fixture "
                f"{ft06_fixture.canonical_sha256}"
            )

        # Cross-verify against JSPLib metadata where present (162 shared non-DMU instances)
        jsplib_match = jsplib_by_name.get(inst_id)
        jsplib_verified = False
        if jsplib_match is not None:
            if (
                int(jsplib_match["jobs"]) != expected_j
                or int(jsplib_match["machines"]) != expected_m
            ):
                raise ValueError(f"{inst_id}: JSPLib dimension mismatch")
            jsplib_verified = True
            cross_checked_jsplib += 1

        # Reconcile bounds carefully without fabricating missing values
        opt = item.get("optimum")
        bounds = item.get("bounds")
        if opt is not None:
            opt_int = int(opt)
            if jsplib_match is not None and jsplib_match.get("optimum") is not None:
                if int(jsplib_match["optimum"]) != opt_int:
                    raise ValueError(
                        f"{inst_id}: optimum mismatch SchedulingLab={opt_int} vs "
                        f"JSPLib={jsplib_match['optimum']}"
                    )
            bound_entry = {
                "instance_id": inst_id,
                "status": "proven_optimal",
                "known_optimum": opt_int,
                "best_known_upper_bound": opt_int,
                "lower_bound": opt_int,
                "source": (
                    "SchedulingLab/jsp-instances classical.json & JSPLib instances.json"
                    if jsplib_verified
                    else "SchedulingLab/jsp-instances classical.json (DMU bounds snapshot)"
                ),
                "recorded_at": SNAPSHOT_DATE,
            }
            proven_count += 1
        elif isinstance(bounds, dict) and bounds.get("upper") is not None:
            ub = int(bounds["upper"])
            lb = int(bounds["lower"]) if bounds.get("lower") is not None else None
            bound_entry = {
                "instance_id": inst_id,
                "status": "best_known_open",
                "known_optimum": None,
                "best_known_upper_bound": ub,
                "lower_bound": lb,
                "source": (
                    "SchedulingLab/jsp-instances classical.json & JSPLib instances.json"
                    if jsplib_verified
                    else "SchedulingLab/jsp-instances classical.json (DMU bounds snapshot)"
                ),
                "recorded_at": SNAPSHOT_DATE,
            }
            open_count += 1
        else:
            raise ValueError(f"{inst_id}: missing both optimum and upper bound in source metadata")

        split = "development" if inst_id in TRACK1_DEVELOPMENT_IDS else "confirmatory"
        if split == "development":
            dev_ids.append(inst_id)
        else:
            confirmatory_ids.append(inst_id)

        durations = [p for row in parsed.processing_times for p in row]
        checksum_records.append(
            {
                "instance_id": inst_id,
                "family": family,
                "split": split,
                "raw_relative_path": raw_rel_from_repo,
                "source_url": f"{SCHEDULINGLAB_BASE_URL}/{item['path']}",
                "retrieval_utc": retrieval_utc,
                "license": "MIT (SchedulingLab/jsp-instances)",
                "format_id": parsed.format_id,
                "parser_version": parsed.parser_version,
                "num_jobs": parsed.num_jobs,
                "num_machines": parsed.num_machines,
                "num_operations": parsed.num_operations,
                "total_processing_time": parsed.total_processing_time,
                "min_processing_time": min(durations),
                "max_processing_time": max(durations),
                "raw_sha256": parsed.raw_sha256,
                "canonical_sha256": parsed.canonical_sha256,
                "jsplib_cross_checked": jsplib_verified,
            }
        )
        bound_records.append(bound_entry)

    if len(dev_ids) != 48 or len(confirmatory_ids) != 194:
        raise RuntimeError(
            f"Split counts mismatch: dev={len(dev_ids)}, conf={len(confirmatory_ids)}"
        )

    metadata_doc = {
        "schema_version": "jssp-track1-metadata-v1",
        "snapshot_date": SNAPSHOT_DATE,
        "retrieval_utc": retrieval_utc,
        "source_repository": "https://github.com/SchedulingLab/jsp-instances",
        "source_index_url": SCHEDULINGLAB_CLASSICAL_JSON_URL,
        "license": "MIT",
        "license_url": SCHEDULINGLAB_LICENSE_URL,
        "jsplib_cross_reference_url": JSPLIB_INSTANCES_JSON_URL,
        "jsplib_cross_checked_count": cross_checked_jsplib,
        "total_instances": len(checksum_records),
        "family_counts": {k.upper(): v for k, v in EXPECTED_FAMILY_COUNTS.items()},
        "instances": checksum_records,
    }
    bounds_doc = {
        "schema_version": "jssp-track1-bounds-v1",
        "snapshot_id": f"track1-bks-{SNAPSHOT_DATE}",
        "recorded_at": SNAPSHOT_DATE,
        "source": "SchedulingLab/jsp-instances classical.json cross-checked with tamy0612/JSPLib",
        "total_instances": len(bound_records),
        "proven_optimal_count": proven_count,
        "best_known_open_count": open_count,
        "bounds": bound_records,
    }
    split_doc = {
        "schema_version": "jssp-track1-split-v1",
        "frozen_in": "src/jssp_research/benchmarks/splits.py",
        "total_instances": 242,
        "development_count": len(dev_ids),
        "confirmatory_count": len(confirmatory_ids),
        "development_ids": sorted(dev_ids),
        "confirmatory_ids": sorted(confirmatory_ids),
    }

    for rel_path, doc in (
        (METADATA_REL_PATH, metadata_doc),
        (BOUNDS_SNAPSHOT_REL_PATH, bounds_doc),
        (SPLIT_REL_PATH, split_doc),
    ):
        target = repo_root / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return {
        "total_instances": len(checksum_records),
        "cross_checked_jsplib": cross_checked_jsplib,
        "proven_optimal_count": proven_count,
        "best_known_open_count": open_count,
        "development_count": len(dev_ids),
        "confirmatory_count": len(confirmatory_ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Acquire and verify 242 Track 1 JSSP benchmarks.")
    parser.add_argument("--force", action="store_true", help="Re-download existing files.")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent download threads.")
    args = parser.parse_args()
    summary = acquire_track1_benchmarks(force=args.force, workers=args.workers)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
