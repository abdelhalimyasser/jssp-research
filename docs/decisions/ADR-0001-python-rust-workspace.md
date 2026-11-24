# ADR-0001 — Python/Rust Hybrid Workspace Architecture

- **Status:** Accepted
- **Date:** 2026-11-23 (documented M6 finalization)
- **Context:** The `jssp-research` framework requires deterministic, high-throughput evaluation across thousands of stochastic runs for classical Job-Shop Scheduling (`N5`–`N8` disjunctive-graph neighborhoods, serial SGS decoding, metaheuristics, memetic and Dual-View GAs) alongside declarative TOML experiment configuration, statistical testing, and reproducibility manifests.
- **Decision:**
  - `crates/jssp-core` (pure Rust, zero Python dependencies) is the single authoritative source of truth for JSSP domain models, OBR chromosomes, deterministic Serial SGS decoding (`D01-A`), feasibility validation, $O(L)$ disjunctive-graph move evaluation (`D03-C`), neighborhood adapters, RNG streams (`ChaCha20Rng`), QMC/MWR initializers, local search, metaheuristics, classical GA, memetic GA, and Dual-View GA.
  - `crates/jssp-python` provides a coarse-grained PyO3 (`0.30.1`) extension module (`jssp_research._native`) built with Maturin (`1.16.0`), returning typed telemetry DTOs per run or parallel batch—never per-move callbacks.
  - `src/jssp_research/` and `scripts/` handle TOML configuration validation, experiment matrix expansion, benchmark metadata/splits, run manifests, non-parametric statistical analysis (Friedman, Kendall $W$, Holm-adjusted Wilcoxon, Vargha–Delaney $A_{12}$, Cliff's $\delta$), and summary reports.
- **Consequences:** All search algorithms execute at compiled Rust speed with strict determinism and zero Python interpreter overhead in inner search loops, while research orchestration, validation, and reporting remain transparent in Python.
