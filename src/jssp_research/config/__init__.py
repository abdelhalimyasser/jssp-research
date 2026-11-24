"""Reproducible configuration schema, loading, and validation helpers."""

from __future__ import annotations

from jssp_research.config.loader import (
    DEFAULT_CONFIG_ROOT,
    load_classic_ga_config,
    load_phase_a_matrix,
    load_phase_b_matrix,
    load_phase_c_matrix,
    load_problem_defaults,
    load_toml_dict,
)
from jssp_research.config.schema import (
    ClassicGaConfigModel,
    PhaseAMatrixConfig,
    PhaseAMatrixEntryModel,
    PhaseBMatrixConfig,
    PhaseBMatrixEntryModel,
    PhaseCMatrixConfig,
    PhaseCMatrixEntryModel,
    ProblemDefaultsConfig,
)
from jssp_research.config.validation import (
    validate_phase_a_configuration,
    validate_phase_b_configuration,
    validate_phase_c_configuration,
)

__all__ = [
    "ClassicGaConfigModel",
    "DEFAULT_CONFIG_ROOT",
    "PhaseAMatrixConfig",
    "PhaseAMatrixEntryModel",
    "PhaseBMatrixConfig",
    "PhaseBMatrixEntryModel",
    "PhaseCMatrixConfig",
    "PhaseCMatrixEntryModel",
    "ProblemDefaultsConfig",
    "load_classic_ga_config",
    "load_phase_a_matrix",
    "load_phase_b_matrix",
    "load_phase_c_matrix",
    "load_problem_defaults",
    "load_toml_dict",
    "validate_phase_a_configuration",
    "validate_phase_b_configuration",
    "validate_phase_c_configuration",
]
