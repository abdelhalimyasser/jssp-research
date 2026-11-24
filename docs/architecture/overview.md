# System Architecture Overview

**Status:** Milestone M3 (Core M2 Scheduling + Deterministic Named RNG, Benchmark Catalog/Parsers & Immutable Run Manifests)

---

## 1. Two-Language Architecture & Responsibility Split

The `jssp-research` framework maintains a strict separation between compute-critical scheduling semantics and high-level experimental orchestration:

1. **Rust Workspace (`crates/`):** Sole source of truth for scheduling semantics, representation validation, decoding, feasibility verification, objective evaluation, portable RNG (`ChaCha20Rng` + SHA-256 stream derivation), and (in subsequent milestones) initializers, local search neighbourhoods, metaheuristics, and memetic GA execution.
   - **`crates/jssp-core`:** Pure Rust library pinning `rand_chacha = "=0.10.0"` and `sha2 = "=0.11.1"`. Exposes typed, deterministic APIs operating on exact integer durations (`u64`), checked arithmetic, and domain-separated ChaCha20 RNG substreams.
   - **`crates/jssp-python`:** Thin PyO3 0.30.1 `cdylib` extension crate (`jssp_research._native`) translating coarse-grained calls between Python sequences and Rust types.
2. **Python Package (`src/jssp_research/`):** Populated in M3 with `native/`, `benchmarks/` (536-instance catalog across Tracks 1–5, 3 versioned parsers backed by Rust validation, SHA-256 checksums, stratified splits), `experiments/` (`seeds.py`, `budgets.py`, `registry.py`, `manifests.py`), `results/` (`schema.py`, `writer.py`, `reader.py`), `utils/provenance.py`, and `cli.py`.

---

## 2. Implemented Crate Layout (`crates/jssp-core/src/`)

| Module | Status | Primary Symbols & Responsibility |
|---|---|---|
| `error.rs` | M2 + M3 | `JsspError` enum with explicit variants for instance, OBR, overflow, precedence, machine-overlap, coverage, makespan, and RNG derivation/sampling errors. |
| `model/` | M2 | `JobId`, `MachineId`, `OperationIndex`, `OperationId`, `Operation`, `JsspInstance`, `MachineSequence`, `ScheduledOperation`, `Schedule`. |
| `representation/` | M2 | `ObrChromosome` construction, multiplicity/bounds validation, and `resolve_operations`. |
| `decoder/` | M2 | Canonical deterministic `decode_obr` (`SerialInsertionDecoder` earliest feasible idle-gap insertion) and reference `decode_obr_append_only`. |
| `feasibility/` | M2 | Independent `validate_schedule`, `check_job_precedence`, and `check_machine_exclusivity`. (`disjunctive_graph.rs` remains deferred to M5.) |
| `objective/` | M2 | Independent pure integer `evaluate_makespan`. |
| `rng/` | **M3** | Portable `ChaCha20Rng` (`rand_chacha-0.10.0`), binary SHA-256 domain separation, `FROZEN_MASTER_SEEDS_V1`, 9 `D07-A` `RngStreamRole`s, `PairingPolicy`, unbiased `gen_range_u64`, Fisher–Yates `shuffle`, and sampling without replacement. |

---

## 3. Core M2 Data Flow

```text
  (jobs: Vec<Vec<(machine, duration)>>, chromosome: Vec<usize>)
        │
        ▼
  JsspInstance::new(...)          ── validates J>=1, M>=1, rectangularity, machine permutations, p>0, u64 bounds
        │
        ▼
  ObrChromosome::resolve_operations(&instance)  ── validates L = J*M, j < J, count[j] == M; maps (j, k)
        │
        ▼
  decoder::decode_obr(&instance, &obr)          ── left-to-right Serial SGS + earliest feasible idle-gap scan
        │
        ▼
  Schedule { operations, makespan }
        │
        ├──► feasibility::validate_schedule(&instance, &schedule)  ── coverage, durations, job pred, machine exclusivity, C_max
        └──► objective::evaluate_makespan(&instance, &schedule)    ── independent max_{j,k} C_{j,k}
```

---

## 4. Planned Milestones Beyond M3

- **M4:** Phase A initial population generators (MWR, Halton, Sobol, scrambled/mixed QMC) and Rust evaluation telemetry counters.
- **M5:** Disjunctive graph critical-path/block analysis, inverse schedule-to-OBR adapter (`D01-B`), `N5`–`N8` neighbourhoods, local search, ILS, SA, and ILS+MH (`Phase B`).
- **M6–M7:** Memetic GA variants (`LGA`, `BGA`, `PLGA`, `Dual-View` Fixed/Adaptive gates in `Phase C`), TOML configuration matrix runner, and statistical analysis/reporting.
