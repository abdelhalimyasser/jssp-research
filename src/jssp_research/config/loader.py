"""TOML configuration loaders using Python 3.11+ stdlib `tomllib`."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from jssp_research.config.schema import (
    ClassicGaConfigModel,
    PhaseAMatrixConfig,
    PhaseBMatrixConfig,
    PhaseCMatrixConfig,
    ProblemDefaultsConfig,
)

DEFAULT_CONFIG_ROOT = Path(__file__).resolve().parents[3] / "configs"


def load_toml_dict(path: Path) -> dict[str, Any]:
    """Read and parse a TOML file strictly as UTF-8."""
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Expected TOML mapping in {path}")
    return data


def load_problem_defaults(config_root: Path = DEFAULT_CONFIG_ROOT) -> ProblemDefaultsConfig:
    """Load `configs/defaults/problem.toml`."""
    return ProblemDefaultsConfig.model_validate(
        load_toml_dict(config_root / "defaults" / "problem.toml")
    )


def load_classic_ga_config(config_root: Path = DEFAULT_CONFIG_ROOT) -> ClassicGaConfigModel:
    """Load `configs/phase_c/classic_ga.toml`."""
    return ClassicGaConfigModel.model_validate(
        load_toml_dict(config_root / "phase_c" / "classic_ga.toml")
    )


def load_phase_a_matrix(config_root: Path = DEFAULT_CONFIG_ROOT) -> PhaseAMatrixConfig:
    """Load `configs/matrix/phase_a_matrix.toml`."""
    return PhaseAMatrixConfig.model_validate(
        load_toml_dict(config_root / "matrix" / "phase_a_matrix.toml")
    )


def load_phase_b_matrix(config_root: Path = DEFAULT_CONFIG_ROOT) -> PhaseBMatrixConfig:
    """Load `configs/matrix/phase_b_matrix.toml`."""
    return PhaseBMatrixConfig.model_validate(
        load_toml_dict(config_root / "matrix" / "phase_b_matrix.toml")
    )


def load_phase_c_matrix(config_root: Path = DEFAULT_CONFIG_ROOT) -> PhaseCMatrixConfig:
    """Load `configs/matrix/phase_c_matrix.toml`."""
    return PhaseCMatrixConfig.model_validate(
        load_toml_dict(config_root / "matrix" / "phase_c_matrix.toml")
    )
