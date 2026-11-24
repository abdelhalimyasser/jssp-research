"""Append-only result writer for per-attempt JSON execution records."""

from __future__ import annotations

import json
from pathlib import Path

from jssp_research.results.schema import RunResultRecord


def default_result_filename(run_id: str, attempt_index: int) -> str:
    """Return canonical immutable filename `{run_id}.attempt-{attempt_index:03d}.result.json`."""
    return f"{run_id}.attempt-{attempt_index:03d}.result.json"


def write_result_record(
    record: RunResultRecord,
    directory_or_file: Path,
) -> Path:
    """Persist a `RunResultRecord` without overwriting prior attempts."""
    if directory_or_file.suffix == ".json":
        target = directory_or_file
    else:
        target = directory_or_file / default_result_filename(record.run_id, record.attempt_index)

    if target.exists():
        raise FileExistsError(f"Result record already exists and cannot be overwritten: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(record.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    target.write_text(serialized, encoding="utf-8")
    return target
