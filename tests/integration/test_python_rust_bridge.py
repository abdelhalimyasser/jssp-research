"""M1-A and M2 integration tests verifying the PyO3/Maturin bridge and error translation."""

from __future__ import annotations

from pathlib import Path

import pytest

import jssp_research
from jssp_research import _native
from jssp_research.native import (
    get_bridge_info,
    get_core_version,
    get_native_bridge_info,
    validate_instance,
    validate_obr,
)


def test_native_module_origin_and_diagnostics() -> None:
    """Verify `_native` loads from the current project build and returns expected IDs."""
    project_root = Path(__file__).resolve().parents[2]
    assert _native.__file__ is not None
    native_path = Path(_native.__file__).resolve()
    assert native_path.is_relative_to(project_root), (
        f"Expected native extension inside {project_root}, got {native_path}"
    )

    assert _native.__version__ == "0.1.0"
    assert jssp_research.__version__ == "0.1.0"
    assert _native.core_version() == "0.1.0"
    assert _native.bridge_info() == "jssp-core/0.1.0"
    assert get_core_version() == "0.1.0"
    assert get_bridge_info() == "jssp-core/0.1.0"
    assert get_native_bridge_info() == {
        "bridge_info": "jssp-core/0.1.0",
        "core_version": "0.1.0",
        "module_version": "0.1.0",
    }


def test_ffi_validation_and_error_translation() -> None:
    """Verify Rust domain errors map cleanly to Python ValueError / OverflowError."""
    valid_jobs = [
        [(0, 2), (1, 1)],
        [(1, 3), (0, 2)],
    ]
    assert validate_instance(valid_jobs) == (2, 2, 4, 8)
    assert validate_obr(valid_jobs, [0, 1, 1, 0]) is True

    with pytest.raises(ValueError, match="at least one job"):
        validate_instance([])

    with pytest.raises(ValueError, match="non-positive processing time"):
        validate_instance([[(0, 0)]])

    with pytest.raises(ValueError, match="OBR chromosome length"):
        validate_obr(valid_jobs, [0, 1])

    with pytest.raises(ValueError, match="invalid job ID"):
        validate_obr(valid_jobs, [0, 1, 2, 0])

    with pytest.raises(OverflowError, match="overflowed"):
        validate_instance([[(0, (1 << 64) - 1), (1, 1)], [(0, 1), (1, 1)]])
