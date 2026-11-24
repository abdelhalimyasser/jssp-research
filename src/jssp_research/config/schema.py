"""TOML configuration schemas for Phase A/B/C defaults, matrices, and protocols."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, model_validator


class ProblemDefaultsConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str
    problem_type: str
    objective: str
    time_representation: str
    allow_zero_duration_operation: bool
    require_positive_job_work: bool
    machine_capacity: int
    allow_preemption: bool
    rectangular_routing_required: bool


class ClassicGaConfigModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str
    configuration_id: str
    name: str
    population_size: int
    tournament_size: int
    selection_tie_break: str
    crossover_operator: str
    crossover_rate: float
    pox_nonempty_proper_partition: bool
    mutation_operator: str
    mutation_rate: float
    elitism_count: int
    elitism_tie_break: str
    budget_formula: str
    budget_min: int
    budget_max: int
    budget_ops_multiplier: float
    step_granularity: int
    streams: dict[str, str]


class PhaseAMatrixEntryModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    config_id: str
    name: str
    family: str
    generator: str
    scrambled: bool
    deterministic: bool
    uses_sobol: bool
    dyadic_net_balance: bool
    alpha: float | None = None
    halton_count: int = 0
    sobol_count: int = 0
    qmc_index_start: int | None = None
    notes: str | None = None
    role: str

    @model_validator(mode="after")
    def _validate_entry(self) -> PhaseAMatrixEntryModel:
        if not (self.config_id.startswith("A") and len(self.config_id) == 3):
            raise ValueError(f"Invalid Phase A config_id: {self.config_id!r}")
        if self.generator == "mixed_halton_sobol":
            if self.halton_count + self.sobol_count != 64:
                raise ValueError(
                    f"{self.config_id}: mixed counts must sum to 64, "
                    f"got {self.halton_count} + {self.sobol_count}"
                )
        return self


class PhaseAMatrixConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str
    phase: str
    expected_config_count: int
    baseline_ga_id: str
    configs: tuple[PhaseAMatrixEntryModel, ...]


class PhaseBMatrixEntryModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    config_id: str
    name: str
    strategy: str
    strategy_code: str
    neighborhood: str
    stochastic: bool
    uses_perturbation: bool

    @model_validator(mode="after")
    def _validate_entry(self) -> PhaseBMatrixEntryModel:
        if not (self.config_id.startswith("B") and len(self.config_id) == 3):
            raise ValueError(f"Invalid Phase B config_id: {self.config_id!r}")
        if self.config_id in {"B25", "B26", "B27", "B28"}:
            raise ValueError(f"{self.config_id} is excluded per D04-A")
        if self.neighborhood not in {"N5", "N6", "N7", "N8"}:
            raise ValueError(f"{self.config_id}: invalid neighborhood {self.neighborhood!r}")
        return self


class PhaseBMatrixConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str
    phase: str
    expected_config_count: int
    excluded_config_ids: tuple[str, ...]
    exclusion_decision: str
    screening_budget_formula: str
    screening_budget_multiplier: int
    confirmatory_budget_formula: str
    confirmatory_budget_multiplier: int
    pairing_mode: str
    pairing_group: str
    configs: tuple[PhaseBMatrixEntryModel, ...]


class PhaseCMatrixEntryModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    config_id: str
    name: str
    architecture: str
    is_memetic: bool
    is_dual_view: bool
    fixed_gate_p: float | None = None
    use_adaptive_gate: bool = False
    w_perf: float | None = None
    w_div: float | None = None
    migration_enabled: bool = False

    @model_validator(mode="after")
    def _validate_entry(self) -> PhaseCMatrixEntryModel:
        if not (self.config_id.startswith("C") and len(self.config_id) == 3):
            raise ValueError(f"Invalid Phase C config_id: {self.config_id!r}")
        if self.w_perf is not None and self.w_div is not None:
            if abs((self.w_perf + self.w_div) - 1.0) > 1e-9:
                raise ValueError(f"{self.config_id}: adaptive weights must sum to 1.0")
        return self


class PhaseCMatrixConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str
    phase: str
    expected_config_count: int
    genotype_capacity: int
    local_search_cap_b_ls: int
    pairing_mode: str
    pairing_group: str
    configs: tuple[PhaseCMatrixEntryModel, ...]
