"""Experiment matrix verification and benchmark-suite construction helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from jssp_research.config import (
    DEFAULT_CONFIG_ROOT,
    PhaseAMatrixEntryModel,
    PhaseBMatrixEntryModel,
    PhaseCMatrixEntryModel,
    load_phase_a_matrix,
    load_phase_b_matrix,
    load_phase_c_matrix,
    validate_phase_a_configuration,
    validate_phase_b_configuration,
    validate_phase_c_configuration,
)


@dataclass(frozen=True, slots=True)
class MatrixInventory:
    phase_a_count: int
    phase_a_ids: tuple[str, ...]
    phase_b_count: int = 28
    phase_b_ids: tuple[str, ...] = ()
    phase_c_count: int = 11
    phase_c_ids: tuple[str, ...] = ()
    expected_total_configurations: int = 63

    @property
    def total_active_configurations(self) -> int:
        return self.phase_a_count + self.phase_b_count + self.phase_c_count


def build_phase_a_matrix(
    config_root: Path = DEFAULT_CONFIG_ROOT,
) -> tuple[PhaseAMatrixEntryModel, ...]:
    """Validate and return ordered Phase A configuration entries."""
    errors = validate_phase_a_configuration(config_root)
    if errors:
        raise ValueError("Invalid Phase A configuration matrix: " + "; ".join(errors))
    return load_phase_a_matrix(config_root).configs


def build_phase_b_matrix(
    config_root: Path = DEFAULT_CONFIG_ROOT,
) -> tuple[PhaseBMatrixEntryModel, ...]:
    """Validate and return ordered active Phase B configuration entries."""
    errors = validate_phase_b_configuration(config_root)
    if errors:
        raise ValueError("Invalid Phase B configuration matrix: " + "; ".join(errors))
    return load_phase_b_matrix(config_root).configs


def build_phase_c_matrix(
    config_root: Path = DEFAULT_CONFIG_ROOT,
) -> tuple[PhaseCMatrixEntryModel, ...]:
    """Validate and return ordered Phase C configuration entries."""
    errors = validate_phase_c_configuration(config_root)
    if errors:
        raise ValueError("Invalid Phase C configuration matrix: " + "; ".join(errors))
    return load_phase_c_matrix(config_root).configs


def summarize_matrix_inventory(config_root: Path = DEFAULT_CONFIG_ROOT) -> MatrixInventory:
    """Return validated configuration IDs and counts across all 63 active configurations."""
    a_entries = build_phase_a_matrix(config_root)
    b_entries = build_phase_b_matrix(config_root)
    c_entries = build_phase_c_matrix(config_root)
    return MatrixInventory(
        phase_a_count=len(a_entries),
        phase_a_ids=tuple(entry.config_id for entry in a_entries),
        phase_b_count=len(b_entries),
        phase_b_ids=tuple(entry.config_id for entry in b_entries),
        phase_c_count=len(c_entries),
        phase_c_ids=tuple(entry.config_id for entry in c_entries),
    )
