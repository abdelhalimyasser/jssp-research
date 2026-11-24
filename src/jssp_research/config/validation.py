"""Cross-file configuration validation against Rust core metadata and frozen plans."""

from __future__ import annotations

from pathlib import Path

from jssp_research.config.loader import (
    DEFAULT_CONFIG_ROOT,
    load_classic_ga_config,
    load_phase_a_matrix,
    load_phase_b_matrix,
    load_phase_c_matrix,
    load_problem_defaults,
    load_toml_dict,
)
from jssp_research.native import (
    list_phase_a_initializers,
    list_phase_b_configs,
    list_phase_c_configs,
)


def validate_phase_a_configuration(config_root: Path = DEFAULT_CONFIG_ROOT) -> list[str]:
    """Validate Phase A TOML files against authoritative Rust `InitializerId` metadata."""
    errors: list[str] = []
    defaults = load_problem_defaults(config_root)
    if defaults.objective != "makespan":
        errors.append(f"Unexpected primary objective: {defaults.objective!r}")

    ga = load_classic_ga_config(config_root)
    if ga.population_size != 64:
        errors.append(f"Classic GA population_size must be 64, got {ga.population_size}")
    if ga.elitism_count != 2:
        errors.append(f"Classic GA elitism_count must be 2, got {ga.elitism_count}")

    matrix = load_phase_a_matrix(config_root)
    if len(matrix.configs) != matrix.expected_config_count or len(matrix.configs) != 24:
        errors.append(
            f"Expected 24 Phase A configs, found {len(matrix.configs)} "
            f"(expected={matrix.expected_config_count})"
        )

    ids = [cfg.config_id for cfg in matrix.configs]
    if len(set(ids)) != len(ids):
        errors.append("Duplicate config_id found in phase_a_matrix.toml")

    rust_specs = {spec.code: spec for spec in list_phase_a_initializers()}
    if set(ids) != set(rust_specs):
        errors.append(f"Phase A IDs mismatch Rust: toml={set(ids)} vs rust={set(rust_specs)}")
        return errors

    for cfg in matrix.configs:
        rspec = rust_specs[cfg.config_id]
        if cfg.deterministic != rspec.is_deterministic:
            errors.append(
                f"{cfg.config_id}: deterministic={cfg.deterministic} "
                f"!= rust {rspec.is_deterministic}"
            )
        if cfg.uses_sobol != rspec.uses_sobol:
            errors.append(
                f"{cfg.config_id}: uses_sobol={cfg.uses_sobol} != rust {rspec.uses_sobol}"
            )
        if cfg.dyadic_net_balance != rspec.has_dyadic_sobol_net_balance:
            errors.append(
                f"{cfg.config_id}: dyadic_net_balance={cfg.dyadic_net_balance} "
                f"!= rust {rspec.has_dyadic_sobol_net_balance}"
            )
        if cfg.alpha != rspec.alpha:
            errors.append(f"{cfg.config_id}: alpha={cfg.alpha} != rust {rspec.alpha}")

    return errors


def validate_phase_b_configuration(config_root: Path = DEFAULT_CONFIG_ROOT) -> list[str]:
    """Validate Phase B TOML files and matrix against Rust `PhaseBConfigId` metadata."""
    errors: list[str] = []
    matrix = load_phase_b_matrix(config_root)
    if len(matrix.configs) != matrix.expected_config_count or len(matrix.configs) != 28:
        errors.append(
            f"Expected 28 active Phase B configs, found {len(matrix.configs)} "
            f"(expected={matrix.expected_config_count})"
        )
    if list(matrix.excluded_config_ids) != ["B25", "B26", "B27", "B28"]:
        errors.append(f"Unexpected Phase B excluded IDs: {matrix.excluded_config_ids!r}")

    mh_doc = load_toml_dict(config_root / "phase_b" / "metropolis_hastings.toml")
    if mh_doc.get("status") != "excluded_per_d04_a":
        errors.append(
            f"Expected status='excluded_per_d04_a' in metropolis_hastings.toml, got {mh_doc}"
        )

    ids = [cfg.config_id for cfg in matrix.configs]
    if len(set(ids)) != len(ids):
        errors.append("Duplicate config_id found in phase_b_matrix.toml")

    rust_specs = {spec.code: spec for spec in list_phase_b_configs()}
    if set(ids) != set(rust_specs):
        errors.append(f"Phase B IDs mismatch Rust: toml={set(ids)} vs rust={set(rust_specs)}")
        return errors

    for cfg in matrix.configs:
        rspec = rust_specs[cfg.config_id]
        if cfg.strategy != rspec.strategy_name:
            errors.append(
                f"{cfg.config_id}: strategy={cfg.strategy!r} != rust {rspec.strategy_name!r}"
            )
        if cfg.neighborhood != rspec.neighborhood_code:
            errors.append(
                f"{cfg.config_id}: neighborhood={cfg.neighborhood!r} != "
                f"rust {rspec.neighborhood_code!r}"
            )

    return errors


def validate_phase_c_configuration(config_root: Path = DEFAULT_CONFIG_ROOT) -> list[str]:
    """Validate Phase C TOML files and matrix against Rust `PhaseCConfigId` metadata."""
    errors: list[str] = []
    matrix = load_phase_c_matrix(config_root)
    if len(matrix.configs) != matrix.expected_config_count or len(matrix.configs) != 11:
        errors.append(
            f"Expected 11 Phase C configs, found {len(matrix.configs)} "
            f"(expected={matrix.expected_config_count})"
        )
    if matrix.genotype_capacity != 64:
        errors.append(f"Expected Phase C genotype_capacity=64, got {matrix.genotype_capacity}")
    if matrix.local_search_cap_b_ls != 50:
        errors.append(
            f"Expected Phase C local_search_cap_b_ls=50, got {matrix.local_search_cap_b_ls}"
        )

    mem_doc = load_toml_dict(config_root / "phase_c" / "memetic_ga.toml")
    if mem_doc.get("local_search_cap_per_individual") != 50:
        errors.append(f"Unexpected memetic_ga.toml b_LS: {mem_doc}")

    dv_doc = load_toml_dict(config_root / "phase_c" / "dual_view.toml")
    if dv_doc.get("paired_individuals") != 32:
        errors.append(f"Unexpected dual_view.toml paired_individuals: {dv_doc}")

    ids = [cfg.config_id for cfg in matrix.configs]
    if len(set(ids)) != len(ids):
        errors.append("Duplicate config_id found in phase_c_matrix.toml")

    rust_specs = {spec.code: spec for spec in list_phase_c_configs()}
    if set(ids) != set(rust_specs):
        errors.append(f"Phase C IDs mismatch Rust: toml={set(ids)} vs rust={set(rust_specs)}")
        return errors

    for cfg in matrix.configs:
        rspec = rust_specs[cfg.config_id]
        if cfg.is_memetic != rspec.is_memetic:
            errors.append(
                f"{cfg.config_id}: is_memetic={cfg.is_memetic} != rust {rspec.is_memetic}"
            )
        if cfg.is_dual_view != rspec.is_dual_view_paired:
            errors.append(
                f"{cfg.config_id}: is_dual_view={cfg.is_dual_view} != "
                f"rust {rspec.is_dual_view_paired}"
            )

    return errors
