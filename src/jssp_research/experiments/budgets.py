"""Computational budget specification and parity validation for run manifests."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)


def _freeze_budget_map(raw: Mapping[str, int]) -> Mapping[str, int]:
    return MappingProxyType(
        {str(k): int(v) for k, v in sorted(raw.items(), key=lambda kv: str(kv[0]))}
    )


class BudgetSpec(BaseModel):
    """Normalized, deeply immutable computational budget record embedded into run manifests."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    objective_evaluation_budget: int
    wall_clock_limit_seconds: float | None = None
    population_size: int | None = None
    iteration_or_generation_budget: int | None = None
    local_search_evaluation_allocation: int | None = None
    algorithm_specific_budgets: Mapping[str, int] = Field(
        default_factory=lambda: MappingProxyType({})
    )

    @field_validator("algorithm_specific_budgets", mode="after")
    @classmethod
    def _freeze_budgets_map(cls, value: Mapping[str, int]) -> Mapping[str, int]:
        return _freeze_budget_map(value)

    @field_serializer("algorithm_specific_budgets")
    def _serialize_budgets_map(self, value: Mapping[str, Any]) -> dict[str, int]:
        return {str(k): int(v) for k, v in value.items()}

    @model_validator(mode="after")
    def _validate_positive_budgets(self) -> BudgetSpec:
        if self.objective_evaluation_budget <= 0:
            raise ValueError("objective_evaluation_budget must be strictly positive")
        if self.wall_clock_limit_seconds is not None and self.wall_clock_limit_seconds <= 0.0:
            raise ValueError("wall_clock_limit_seconds must be strictly positive when provided")
        if self.population_size is not None and self.population_size <= 0:
            raise ValueError("population_size must be strictly positive when provided")
        if (
            self.iteration_or_generation_budget is not None
            and self.iteration_or_generation_budget <= 0
        ):
            raise ValueError("iteration_or_generation_budget must be strictly positive")
        if (
            self.local_search_evaluation_allocation is not None
            and self.local_search_evaluation_allocation < 0
        ):
            raise ValueError("local_search_evaluation_allocation cannot be negative")
        for key, value in self.algorithm_specific_budgets.items():
            if value < 0:
                raise ValueError(f"algorithm_specific_budgets[{key!r}] cannot be negative")
        return self
