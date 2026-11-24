"""Tests for TOML configuration loading and validation against Rust metadata."""

from __future__ import annotations

from jssp_research.config import (
    load_classic_ga_config,
    load_phase_a_matrix,
    load_phase_b_matrix,
    load_phase_c_matrix,
    load_problem_defaults,
    validate_phase_a_configuration,
    validate_phase_b_configuration,
    validate_phase_c_configuration,
)


def test_phase_a_toml_configs_validate_cleanly() -> None:
    """Verify problem defaults, C01 GA config, and 24-config Phase A matrix."""
    errors = validate_phase_a_configuration()
    assert errors == []

    defaults = load_problem_defaults()
    assert defaults.objective == "makespan"
    assert defaults.allow_zero_duration_operation is True
    assert defaults.require_positive_job_work is True

    ga = load_classic_ga_config()
    assert ga.population_size == 64
    assert ga.crossover_rate == 0.85
    assert ga.mutation_rate == 0.15
    assert ga.elitism_count == 2

    matrix = load_phase_a_matrix()
    assert len(matrix.configs) == 24
    by_id = {c.config_id: c for c in matrix.configs}
    assert by_id["A12"].halton_count == 32 and by_id["A12"].sobol_count == 32
    assert by_id["A13"].dyadic_net_balance is False
    assert by_id["A14"].halton_count == 48 and by_id["A14"].sobol_count == 16


def test_phase_b_toml_configs_validate_cleanly() -> None:
    """Verify active 28-config Phase B matrix and D04-A exclusion note."""
    errors = validate_phase_b_configuration()
    assert errors == []

    matrix_b = load_phase_b_matrix()
    assert len(matrix_b.configs) == 28
    assert list(matrix_b.excluded_config_ids) == ["B25", "B26", "B27", "B28"]
    assert matrix_b.screening_budget_multiplier == 100
    assert matrix_b.confirmatory_budget_multiplier == 500


def test_phase_c_toml_configs_validate_cleanly() -> None:
    """Verify 11-config Phase C matrix and Model A Dual-View parameters."""
    errors = validate_phase_c_configuration()
    assert errors == []

    matrix_c = load_phase_c_matrix()
    assert len(matrix_c.configs) == 11
    assert matrix_c.genotype_capacity == 64
    assert matrix_c.local_search_cap_b_ls == 50
