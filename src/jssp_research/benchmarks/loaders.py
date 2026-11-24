"""Versioned classical JSSP file parsers backed by Rust domain validation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from jssp_research.benchmarks.checksums import canonical_instance_sha256, sha256_bytes
from jssp_research.native import validate_instance


class BenchmarkParseError(ValueError):
    """Raised when a benchmark instance file violates format or JSSP invariants."""

    def __init__(self, message: str, *, source: str = "<memory>", line: int | None = None) -> None:
        location = f"{source}:{line}" if line is not None else source
        super().__init__(f"{location}: {message}")
        self.source = source
        self.line = line
        self.detail = message


class UnsupportedBenchmarkFormatError(ValueError):
    """Raised when an unrecognised benchmark format identifier is requested."""


@dataclass(frozen=True, slots=True)
class ParserSpec:
    """Metadata describing a registered benchmark format parser."""

    format_id: str
    parser_version: str
    description: str
    parser: Callable[[str, str], tuple[int, int, tuple[tuple[tuple[int, int], ...], ...]]]


@dataclass(frozen=True, slots=True)
class ParsedJsspInstance:
    """Canonical validated classical rectangular JSSP instance returned by parsers."""

    instance_id: str
    num_jobs: int
    num_machines: int
    num_operations: int
    total_processing_time: int
    jobs: tuple[tuple[tuple[int, int], ...], ...]
    machine_routes: tuple[tuple[int, ...], ...]
    processing_times: tuple[tuple[int, ...], ...]
    format_id: str
    parser_version: str
    raw_sha256: str
    canonical_sha256: str


def _parse_int_tokens(
    raw_tokens: list[str],
    *,
    source: str,
    line_no: int,
) -> list[int]:
    values: list[int] = []
    for tok in raw_tokens:
        try:
            values.append(int(tok))
        except ValueError as exc:
            raise BenchmarkParseError(
                f"expected integer token, got {tok!r}", source=source, line=line_no
            ) from exc
    return values


def _non_comment_lines(text: str) -> list[tuple[int, str]]:
    lines: list[tuple[int, str]] = []
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("//"):
            continue
        lines.append((line_no, stripped))
    return lines


def _parse_interleaved_jssp(
    text: str,
    source: str,
    *,
    machine_index_base: int,
) -> tuple[int, int, tuple[tuple[tuple[int, int], ...], ...]]:
    lines = _non_comment_lines(text)
    if not lines:
        raise BenchmarkParseError("benchmark file is empty or comments only", source=source, line=1)

    header_line_no, header_text = lines[0]
    header_ints = _parse_int_tokens(header_text.split(), source=source, line_no=header_line_no)
    if len(header_ints) != 2:
        raise BenchmarkParseError(
            f"header must contain 2 integers '<num_jobs> <num_machines>', got {len(header_ints)}",
            source=source,
            line=header_line_no,
        )

    num_jobs, num_machines = header_ints
    if num_jobs <= 0 or num_machines <= 0:
        raise BenchmarkParseError(
            f"instance dimensions must be positive, got {num_jobs}x{num_machines}",
            source=source,
            line=header_line_no,
        )

    body = lines[1:]
    if len(body) != num_jobs:
        raise BenchmarkParseError(
            f"expected {num_jobs} job rows, found {len(body)}",
            source=source,
            line=header_line_no,
        )

    parsed_jobs: list[tuple[tuple[int, int], ...]] = []
    expected_tokens = 2 * num_machines
    for job_idx, (line_no, row_text) in enumerate(body):
        ints = _parse_int_tokens(row_text.split(), source=source, line_no=line_no)
        if len(ints) != expected_tokens:
            raise BenchmarkParseError(
                f"job {job_idx} row must contain {expected_tokens} integers "
                f"(2 * {num_machines} operations), got {len(ints)}",
                source=source,
                line=line_no,
            )

        ops: list[tuple[int, int]] = []
        for op_idx in range(num_machines):
            raw_machine = ints[2 * op_idx]
            duration = ints[2 * op_idx + 1]
            if machine_index_base == 1:
                if raw_machine < 1:
                    raise BenchmarkParseError(
                        f"op ({job_idx}, {op_idx}) has invalid 1-based machine {raw_machine}",
                        source=source,
                        line=line_no,
                    )
                machine = raw_machine - 1
            else:
                if raw_machine < 0:
                    raise BenchmarkParseError(
                        f"operation ({job_idx}, {op_idx}) has negative machine ID {raw_machine}",
                        source=source,
                        line=line_no,
                    )
                machine = raw_machine

            if duration <= 0:
                raise BenchmarkParseError(
                    f"operation ({job_idx}, {op_idx}) has non-positive processing time {duration}",
                    source=source,
                    line=line_no,
                )
            ops.append((machine, duration))
        parsed_jobs.append(tuple(ops))

    return num_jobs, num_machines, tuple(parsed_jobs)


def _parse_standard_0based_v1(
    text: str,
    source: str,
) -> tuple[int, int, tuple[tuple[tuple[int, int], ...], ...]]:
    return _parse_interleaved_jssp(text, source, machine_index_base=0)


def _parse_standard_1based_v1(
    text: str,
    source: str,
) -> tuple[int, int, tuple[tuple[tuple[int, int], ...], ...]]:
    return _parse_interleaved_jssp(text, source, machine_index_base=1)


def _parse_taillard_matrix_v1(
    text: str,
    source: str,
) -> tuple[int, int, tuple[tuple[tuple[int, int], ...], ...]]:
    """Parse Taillard (1993) dual-matrix format (`Times` block followed by 1-based `Machines`)."""
    lines = _non_comment_lines(text)
    if not lines:
        raise BenchmarkParseError("Taillard benchmark input is empty", source=source, line=1)

    idx = 0
    first_line_no, first_text = lines[idx]
    header_ints = _parse_int_tokens(first_text.split(), source=source, line_no=first_line_no)
    if len(header_ints) < 2:
        raise BenchmarkParseError(
            "Taillard header must start with '<num_jobs> <num_machines> ...'",
            source=source,
            line=first_line_no,
        )
    num_jobs, num_machines = header_ints[0], header_ints[1]
    if num_jobs <= 0 or num_machines <= 0:
        raise BenchmarkParseError(
            f"Taillard dimensions must be positive, got {num_jobs}x{num_machines}",
            source=source,
            line=first_line_no,
        )
    idx += 1

    if idx < len(lines) and lines[idx][1].lower() == "times":
        idx += 1

    if idx + num_jobs > len(lines):
        raise BenchmarkParseError(
            f"incomplete Taillard Times matrix: expected {num_jobs} rows",
            source=source,
            line=lines[-1][0],
        )

    times_matrix: list[list[int]] = []
    for job_idx in range(num_jobs):
        line_no, row_text = lines[idx + job_idx]
        row = _parse_int_tokens(row_text.split(), source=source, line_no=line_no)
        if len(row) != num_machines:
            raise BenchmarkParseError(
                f"Taillard Times row {job_idx} must have {num_machines} integers, got {len(row)}",
                source=source,
                line=line_no,
            )
        times_matrix.append(row)
    idx += num_jobs

    if idx < len(lines) and lines[idx][1].lower() == "machines":
        idx += 1

    if idx + num_jobs != len(lines):
        raise BenchmarkParseError(
            f"Taillard Machines matrix must contain {num_jobs} rows, found {len(lines) - idx}",
            source=source,
            line=lines[idx][0] if idx < len(lines) else lines[-1][0],
        )

    machines_matrix: list[list[int]] = []
    for job_idx in range(num_jobs):
        line_no, row_text = lines[idx + job_idx]
        row = _parse_int_tokens(row_text.split(), source=source, line_no=line_no)
        if len(row) != num_machines:
            raise BenchmarkParseError(
                f"Taillard Machines row {job_idx} must have {num_machines} ints, got {len(row)}",
                source=source,
                line=line_no,
            )
        for op_idx, machine_1based in enumerate(row):
            if machine_1based < 1:
                raise BenchmarkParseError(
                    f"Taillard machine at ({job_idx}, {op_idx}) must be >= 1, got {machine_1based}",
                    source=source,
                    line=line_no,
                )
        machines_matrix.append([m - 1 for m in row])

    jobs = tuple(
        tuple((machines_matrix[j][k], times_matrix[j][k]) for k in range(num_machines))
        for j in range(num_jobs)
    )
    return num_jobs, num_machines, jobs


PARSER_REGISTRY: dict[str, ParserSpec] = {
    "standard_jssp_0based_v1": ParserSpec(
        format_id="standard_jssp_0based_v1",
        parser_version="1.0.0",
        description="Standard JSPLib / OR-Library rectangular format with 0-based machine IDs.",
        parser=_parse_standard_0based_v1,
    ),
    "standard_jssp_1based_v1": ParserSpec(
        format_id="standard_jssp_1based_v1",
        parser_version="1.0.0",
        description="Interleaved rectangular JSSP format with 1-based machine IDs.",
        parser=_parse_standard_1based_v1,
    ),
    "taillard_matrix_v1": ParserSpec(
        format_id="taillard_matrix_v1",
        parser_version="1.0.0",
        description="Taillard (1993) dual-matrix format (`Times` + 1-based `Machines` blocks).",
        parser=_parse_taillard_matrix_v1,
    ),
}


def list_supported_formats() -> tuple[ParserSpec, ...]:
    """Return all registered benchmark format specifications in deterministic ID order."""
    return tuple(PARSER_REGISTRY[key] for key in sorted(PARSER_REGISTRY))


def parse_jssp_text(
    text: str,
    *,
    instance_id: str,
    format_id: str = "standard_jssp_0based_v1",
    source: str = "<memory>",
    raw_bytes: bytes | None = None,
) -> ParsedJsspInstance:
    """Parse raw benchmark text and validate canonical JSSP constraints via Rust `jssp-core`."""
    if format_id not in PARSER_REGISTRY:
        supported = ", ".join(sorted(PARSER_REGISTRY))
        raise UnsupportedBenchmarkFormatError(
            f"Unsupported benchmark format {format_id!r}; expected one of: {supported}"
        )

    spec = PARSER_REGISTRY[format_id]
    num_jobs, num_machines, jobs = spec.parser(text, source)

    try:
        rust_j, rust_m, rust_ops, total_work = validate_instance(jobs)
    except (ValueError, OverflowError) as exc:
        raise BenchmarkParseError(f"Rust domain validation failed: {exc}", source=source) from exc

    if (rust_j, rust_m) != (num_jobs, num_machines):
        raise BenchmarkParseError(
            f"dimension mismatch: header {num_jobs}x{num_machines} vs Rust {rust_j}x{rust_m}",
            source=source,
        )

    payload_bytes = raw_bytes if raw_bytes is not None else text.encode("utf-8")
    routes = tuple(tuple(m for m, _ in job_row) for job_row in jobs)
    durations = tuple(tuple(p for _, p in job_row) for job_row in jobs)

    return ParsedJsspInstance(
        instance_id=instance_id,
        num_jobs=rust_j,
        num_machines=rust_m,
        num_operations=rust_ops,
        total_processing_time=total_work,
        jobs=jobs,
        machine_routes=routes,
        processing_times=durations,
        format_id=spec.format_id,
        parser_version=spec.parser_version,
        raw_sha256=sha256_bytes(payload_bytes),
        canonical_sha256=canonical_instance_sha256(jobs),
    )


def parse_jssp_file(
    path: Path,
    *,
    instance_id: str | None = None,
    format_id: str = "standard_jssp_0based_v1",
) -> ParsedJsspInstance:
    """Read raw file bytes, decode UTF-8 text, parse, and validate with Rust."""
    raw_bytes = path.read_bytes()
    text = raw_bytes.decode("utf-8")
    resolved_id = instance_id if instance_id is not None else path.stem
    return parse_jssp_text(
        text,
        instance_id=resolved_id,
        format_id=format_id,
        source=str(path),
        raw_bytes=raw_bytes,
    )
