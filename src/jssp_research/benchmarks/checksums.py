"""SHA-256 hashing and canonical instance serialization helpers."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from pathlib import Path

_HEX_64_RE = re.compile(r"^[0-9a-f]{64}$")


class ChecksumMismatchError(ValueError):
    """Raised when an actual SHA-256 digest does not match expectations."""


def sha256_bytes(data: bytes) -> str:
    """Return lowercase 64-character hexadecimal SHA-256 digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    """Return lowercase 64-character hexadecimal SHA-256 digest of UTF-8 text."""
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    """Stream and return lowercase 64-character SHA-256 digest of a file on disk."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def canonical_instance_payload(jobs: Sequence[Sequence[tuple[int, int]]]) -> bytes:
    """Encode parsed classical JSSP instance data into deterministic compact UTF-8 JSON."""
    num_jobs = len(jobs)
    num_machines = len(jobs[0]) if jobs else 0
    payload = {
        "num_jobs": int(num_jobs),
        "num_machines": int(num_machines),
        "jobs": [[[int(machine), int(duration)] for machine, duration in row] for row in jobs],
    }
    return json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")


def canonical_instance_sha256(jobs: Sequence[Sequence[tuple[int, int]]]) -> str:
    """Return the canonical SHA-256 checksum of normalized parsed instance data."""
    return sha256_bytes(canonical_instance_payload(jobs))


def is_valid_sha256_hex(digest: str) -> bool:
    """Return True if `digest` is a lowercase 64-character hexadecimal SHA-256 string."""
    return bool(_HEX_64_RE.fullmatch(digest))


def verify_sha256(actual: str, expected: str, *, context: str = "") -> None:
    """Assert `actual` equals `expected` as lowercase 64-char SHA-256 digests."""
    if not is_valid_sha256_hex(actual):
        raise ChecksumMismatchError(
            f"Invalid actual SHA-256 format{' for ' + context if context else ''}: {actual!r}"
        )
    if not is_valid_sha256_hex(expected):
        raise ChecksumMismatchError(
            f"Invalid expected SHA-256 format{' for ' + context if context else ''}: {expected!r}"
        )
    if actual != expected:
        label = f" ({context})" if context else ""
        raise ChecksumMismatchError(f"SHA-256 mismatch{label}: expected {expected}, got {actual}")
