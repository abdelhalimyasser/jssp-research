"""Build the literature-grounded benchmark selection record for Track 1 (242 instances)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from jssp_research.benchmarks.splits import TRACK1_DEVELOPMENT_IDS

REPO_ROOT = Path(__file__).resolve().parents[1]

# Frozen 32-instance Common Sequential Pipeline Suite (shared across Phases A, B, C, Final).
# Stratified identically across all 8 classical Track 1 literature families (`FT`, `LA`, `ABZ`,
# `ORB`, `SWV`, `YN`, `TA`, `DMU`) with strict 16 Development / 16 Held-Out isolation.
COMMON_PIPELINE_DEV_IDS: tuple[str, ...] = (
    "ft06",
    "la01",
    "la06",
    "la11",
    "la16",
    "la21",
    "abz5",
    "orb01",
    "orb06",
    "swv01",
    "swv06",
    "yn1",
    "ta01",
    "ta11",
    "dmu01",
    "dmu06",
)

COMMON_PIPELINE_HELD_OUT_IDS: tuple[str, ...] = (
    "ft10",
    "ft20",
    "la02",
    "la19",
    "la24",
    "la38",
    "abz6",
    "abz7",
    "orb02",
    "orb07",
    "swv02",
    "swv07",
    "yn2",
    "ta02",
    "ta12",
    "dmu02",
)

COMMON_PIPELINE_ALL_IDS: frozenset[str] = frozenset(
    COMMON_PIPELINE_DEV_IDS + COMMON_PIPELINE_HELD_OUT_IDS
)

FAMILY_PUBLISHED_REFS: dict[str, str] = {
    "FT": (
        "Fisher & Thompson (1963); Nowicki & Smutnicki (1996); Balas & Vazacopoulos (1998); "
        "Zhang et al. (2007); Xie et al. (2023); Whitley et al. (1994); El-Mihoub et al. (2021)"
    ),
    "LA": (
        "Lawrence (1984); Nowicki & Smutnicki (1996); Balas & Vazacopoulos (1998); "
        "Zhang et al. (2007); Xie et al. (2023); Gui et al. (2023)"
    ),
    "ABZ": (
        "Adams, Balas & Zawack (1988); Nowicki & Smutnicki (1996); Balas & Vazacopoulos (1998); "
        "Zhang et al. (2007); Xie et al. (2023)"
    ),
    "ORB": (
        "Applegate & Cook (1991); Balas & Vazacopoulos (1998); Zhang et al. (2007); "
        "Xie et al. (2023); Gui et al. (2023)"
    ),
    "SWV": (
        "Storer, Wu & Vaccari (1992); Balas & Vazacopoulos (1998); Zhang et al. (2007); "
        "Xie et al. (2023); Gui et al. (2023)"
    ),
    "YN": (
        "Yamada & Nakano (1992); Balas & Vazacopoulos (1998); Zhang et al. (2007); "
        "Xie et al. (2023)"
    ),
    "TA": (
        "Taillard (1993); Nowicki & Smutnicki (1996); Zhang et al. (2007); "
        "Xie et al. (2023); Gui et al. (2023); Ba et al. (2026)"
    ),
    "DMU": (
        "Demirkol, Mehta & Uzsoy (1998); Kuhpfahl & Bierwirth (2016) [TWT]; "
        "SchedulingLab / JSP-Instances Classical Cmax Track"
    ),
}


def build_selection_records(repo_root: Path = REPO_ROOT) -> list[dict[str, Any]]:
    """Combine verified Track 1 metadata, bounds snapshot, and split with literature refs."""
    if not set(COMMON_PIPELINE_DEV_IDS).issubset(TRACK1_DEVELOPMENT_IDS):
        raise RuntimeError("COMMON_PIPELINE_DEV_IDS must be a subset of TRACK1_DEVELOPMENT_IDS")
    if set(COMMON_PIPELINE_HELD_OUT_IDS) & TRACK1_DEVELOPMENT_IDS:
        raise RuntimeError("COMMON_PIPELINE_HELD_OUT_IDS must be disjoint from development split")
    if len(COMMON_PIPELINE_DEV_IDS) != 16 or len(COMMON_PIPELINE_HELD_OUT_IDS) != 16:
        raise RuntimeError("Expected exact 16 dev + 16 held-out common pipeline suite")

    meta_doc = json.loads(
        (repo_root / "data/metadata/track1_checksums_and_provenance.json").read_text("utf-8")
    )
    bounds_doc = json.loads(
        (repo_root / "data/bounds/track1_bks_snapshot_20261124.json").read_text("utf-8")
    )
    bounds_by_id = {row["instance_id"]: row for row in bounds_doc["bounds"]}

    records: list[dict[str, Any]] = []
    for row in meta_doc["instances"]:
        inst_id = str(row["instance_id"])
        fam = str(row["family"])
        b_row = bounds_by_id[inst_id]
        bks = (
            int(b_row["known_optimum"])
            if b_row.get("known_optimum") is not None
            else int(b_row["best_known_upper_bound"])
        )
        lb = int(b_row["lower_bound"]) if b_row.get("lower_bound") is not None else None
        raw_path = repo_root / str(row["raw_relative_path"])
        verified = "verified_local_file_and_rust_domain" if raw_path.is_file() else "metadata_only"

        records.append(
            {
                "instance_id": inst_id,
                "benchmark_family": fam,
                "split": "development" if inst_id in TRACK1_DEVELOPMENT_IDS else "held_out",
                "in_common_pipeline_suite": inst_id in COMMON_PIPELINE_ALL_IDS,
                "source": "SchedulingLab/jsp-instances"
                + (" & tamy0612/JSPLib" if row["jsplib_cross_checked"] else ""),
                "source_url": str(row["source_url"]),
                "raw_relative_path": str(row["raw_relative_path"]),
                "jobs": int(row["num_jobs"]),
                "machines": int(row["num_machines"]),
                "operation_count": int(row["num_operations"]),
                "published_references": FAMILY_PUBLISHED_REFS[fam],
                "best_known_value": bks,
                "lower_bound": lb,
                "bound_status": str(b_row["status"]),
                "raw_checksum": str(row["raw_sha256"]),
                "canonical_checksum": str(row["canonical_sha256"]),
                "verified_status": verified,
            }
        )
    return records


def main() -> None:
    records = build_selection_records()
    out_json = REPO_ROOT / "data/metadata/selected_benchmark_suite.json"
    out_csv = REPO_ROOT / "data/metadata/selected_benchmark_suite.csv"

    common_dev = [
        r["instance_id"]
        for r in records
        if r["in_common_pipeline_suite"] and r["split"] == "development"
    ]
    common_ho = [
        r["instance_id"]
        for r in records
        if r["in_common_pipeline_suite"] and r["split"] == "held_out"
    ]

    summary = {
        "schema_version": "jssp-selected-benchmark-suite-v1",
        "total_instances": len(records),
        "development_count": sum(1 for r in records if r["split"] == "development"),
        "held_out_count": sum(1 for r in records if r["split"] == "held_out"),
        "common_pipeline_suite_count": len(common_dev) + len(common_ho),
        "common_pipeline_dev_ids": sorted(common_dev),
        "common_pipeline_held_out_ids": sorted(common_ho),
        "proven_optimal_count": sum(1 for r in records if r["bound_status"] == "proven_optimal"),
        "best_known_open_count": sum(1 for r in records if r["bound_status"] == "best_known_open"),
        "families": sorted({r["benchmark_family"] for r in records}),
        "instances": records,
    }
    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    fieldnames = list(records[0].keys())
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(
        f"Wrote {out_json} & {out_csv}: total={summary['total_instances']} "
        f"(dev={summary['development_count']}, held_out={summary['held_out_count']}, "
        f"common_suite={summary['common_pipeline_suite_count']} [16 dev + 16 held_out])"
    )


if __name__ == "__main__":
    main()
