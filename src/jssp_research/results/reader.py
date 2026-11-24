"""Result reader and manifest cross-reference verification helpers."""

from __future__ import annotations

import json
from pathlib import Path

from jssp_research.experiments.manifests import RunManifest
from jssp_research.results.schema import RunResultRecord, validate_result_against_manifest


def load_and_validate_result(
    path: Path,
    *,
    manifest: RunManifest | None = None,
) -> RunResultRecord:
    """Load a result JSON file, validate schema and `result_sha256`, and check manifest link."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    record = RunResultRecord.model_validate(raw)
    if manifest is not None:
        validate_result_against_manifest(manifest, record)
    return record
