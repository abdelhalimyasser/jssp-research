"""Immutable separate execution result records and append-only I/O."""

from __future__ import annotations

from jssp_research.results.reader import load_and_validate_result
from jssp_research.results.schema import (
    RESULT_SCHEMA_VERSION,
    SUPPORTED_RESULT_VERSIONS,
    ExecutionMetricsSection,
    ObjectiveResultSection,
    ResultValidationError,
    RunErrorInfo,
    RunResultRecord,
    RunStatus,
    create_run_result_record,
    validate_result_against_manifest,
)
from jssp_research.results.writer import default_result_filename, write_result_record

__all__ = [
    "ExecutionMetricsSection",
    "ObjectiveResultSection",
    "RESULT_SCHEMA_VERSION",
    "ResultValidationError",
    "RunErrorInfo",
    "RunResultRecord",
    "RunStatus",
    "SUPPORTED_RESULT_VERSIONS",
    "create_run_result_record",
    "default_result_filename",
    "load_and_validate_result",
    "validate_result_against_manifest",
    "write_result_record",
]
