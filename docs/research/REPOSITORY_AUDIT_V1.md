# REPOSITORY_AUDIT_V1 — Milestone M0 Audit Report

**Audit Date:** 2026-11-23
**Repository:** `/home/abdelhalimyasser/Downloads/jssp-research` (`main` @ commit `36564e2aacd312218b56c7413181c55a10ff14a8`)
**Scope:** Full inventory, module classification, toolchain/environment validation, configuration reconciliation, risk register, and implementation roadmap prior to writing production algorithms.

---

## 1. Repository Inventory

Commit `36564e2aacd312218b56c7413181c55a10ff14a8` (`chore: initialize jssp-research workspace structure`) checked in **219 files** (mostly empty 0-byte scaffolds plus `.python-version`). Subsequent local uncommitted edits populated `AGENTS.md`, `README.md`, `pyproject.toml`, and `uv.lock`.

- **Total tracked workspace files:** **219 files** (**218** when excluding root `.gitignore` via `-not -path './.git*'`).
- **Populated (non-empty) files:** **5 files** (`1,876` lines / `270,720` bytes total):
  - [`AGENTS.md`](file:///home/abdelhalimyasser/Downloads/jssp-research/AGENTS.md) — `300` lines / `24,044` bytes (uncommitted modification over 0-byte committed file)
  - [`README.md`](file:///home/abdelhalimyasser/Downloads/jssp-research/README.md) — `684` lines / `38,152` bytes (uncommitted modification over 0-byte committed file)
  - [`pyproject.toml`](file:///home/abdelhalimyasser/Downloads/jssp-research/pyproject.toml) — `61` lines / `1,281` bytes (uncommitted expansion over 5-line committed file)
  - [`.python-version`](file:///home/abdelhalimyasser/Downloads/jssp-research/.python-version) — `1` line / `5` bytes (`3.15`)
  - [`uv.lock`](file:///home/abdelhalimyasser/Downloads/jssp-research/uv.lock) — `830` lines / `207,238` bytes (uncommitted lockfile revision 5, currently stale relative to `pyproject.toml`)
- **Empty placeholder (`0`-byte) files before M0 docs:** **214 files** (`213` excluding root `.gitignore`):

| Directory / Group | Total Files | Populated | 0-Byte Placeholders | Notes |
|---|---:|---:|---:|---|
| Root config & metadata | 13 | 5 | 8 | Empty: `.gitignore`, `CITATION.cff`, `Cargo.lock`, `Cargo.toml`, `LICENSE`, `NOTICE`, `rust-toolchain.toml`, `rustfmt.toml` |
| `benches/` | 5 | 0 | 5 | `benchmark_protocol.md` + 4 Rust Criterion/bench stubs |
| `configs/` | 29 | 0 | 29 | `benchmarks/` (5), `defaults/` (4), `matrix/` (3), `phase_a/` (3), `phase_b/` (7), `phase_c/` (3), `protocols/` (4) |
| `crates/jssp-core/` | 76 | 0 | 76 | `Cargo.toml` (1), `src/` (70 `.rs` files), `tests/` (5 `.rs` files) |
| `crates/jssp-python/` | 6 | 0 | 6 | `Cargo.toml` (1), `src/` (5 `.rs` files) |
| `data/` | 5 | 0 | 5 | 5 `.gitignore` files in `bounds/`, `metadata/`, `processed/`, `raw/`, `splits/` |
| `docs/` (pre-M0) | 15 | 0 | 15 | `algorithms/` (5), `architecture/` (3), `benchmarks/` (2), `decisions/` (1), `experiments/` (4) |
| `paper/` | 5 | 0 | 5 | 5 `.gitignore` files |
| `reports/` | 5 | 0 | 5 | 5 `.gitignore` files |
| `results/` | 5 | 0 | 5 | 5 `.gitignore` files |
| `scripts/` | 6 | 0 | 6 | All 6 orchestration/analysis scripts are 0 bytes |
| `src/jssp_research/` | 39 | 0 | 39 | `__init__.py`, `cli.py`, and 8 subpackages (all 0 bytes) |
| `tests/` | 10 | 0 | 10 | `fixtures/` (3), `integration/` (3), `python/` (4) |
| **Total (pre-M0 docs)** | **219** | **5** | **214** | 0% of code, config, tests, and fixtures implemented |

---

## 2. Current Implementation Status (Module Classification)

Using the required taxonomy (**Implemented**, **Partially implemented**, **Placeholder**, **Missing**, **Implemented but untested**, **Implemented and tested**):

| # | Module / Subsystem | Repository Paths | Status | Verified Evidence |
|---|---|---|---|---|
| 1 | **OBR representation** | `crates/jssp-core/src/representation/{mod,obr}.rs` | **Placeholder** | 2 source files at 0 bytes; test `obr_properties.rs` is 0 bytes |
| 2 | **JSSP instance model** | `crates/jssp-core/src/model/{mod,identifiers,instance,machine_sequence,operation,schedule}.rs` | **Placeholder** | All 6 source files are 0 bytes |
| 3 | **Schedule decoder** | `crates/jssp-core/src/decoder/{mod,serial_insertion}.rs` | **Placeholder** | 2 source files at 0 bytes; test `decoder_correctness.rs` is 0 bytes |
| 4 | **Feasibility validation** | `crates/jssp-core/src/feasibility/{mod,disjunctive_graph,machine_exclusivity,precedence}.rs` | **Placeholder** | All 4 source files at 0 bytes; test `feasibility.rs` is 0 bytes |
| 5 | **Makespan evaluation** | `crates/jssp-core/src/objective/{mod,makespan}.rs` | **Placeholder** | Both source files are 0 bytes |
| 6 | **Critical path & blocks** | `crates/jssp-core/src/graph/{mod,critical_path,critical_blocks}.rs` | **Placeholder** | All 3 source files are 0 bytes |
| 7 | **N5, N6, N7, and N8** | `crates/jssp-core/src/neighborhoods/{mod,moves,n5,n6,n7,n8,validation}.rs` | **Placeholder** | All 7 source files at 0 bytes; test `neighbourhood_round_trip.rs` and `docs/algorithms/n5_n6_n7_n8_definitions.md` are 0 bytes |
| 8 | **Initial population generators** | `crates/jssp-core/src/initialization/{mod,random,mwr,monte_carlo}.rs`, `qmc/{mod,halton,sobol,scrambling,discrete_mapping}.rs` | **Placeholder** | All 9 source files at 0 bytes; `docs/algorithms/qmc_mapping.md` is 0 bytes |
| 9 | **Local search** | `crates/jssp-core/src/local_search/{mod,evaluation}.rs`, `descent/{mod,steepest,first_improvement,random_improvement}.rs` | **Placeholder** | All 6 source files are 0 bytes |
| 10 | **Metaheuristics** | `crates/jssp-core/src/metaheuristics/{mod,tabu_search,ils,simulated_annealing,metropolis_hastings,ils_mh}.rs` | **Placeholder** | All 6 source files at 0 bytes; `docs/algorithms/mh_vs_sa.md` is 0 bytes |
| 11 | **Genetic algorithms** | `crates/jssp-core/src/genetic/{mod,population,selection,crossover,mutation,elitism}.rs` | **Placeholder** | All 6 source files are 0 bytes |
| 12 | **Memetic algorithms** | `crates/jssp-core/src/memetic/{mod,lamarckian,baldwinian,partial_lamarckian}.rs` | **Placeholder** | All 4 source files are 0 bytes |
| 13 | **Dual-View implementation** | `crates/jssp-core/src/memetic/dual_view/{mod,individual,fixed_gate,adaptive_gate,reproduction,migration}.rs` | **Placeholder** | All 6 source files at 0 bytes; `docs/algorithms/dual_view.md` is 0 bytes |
| 14 | **Python–Rust bindings** | `crates/jssp-python/{Cargo.toml,src/*}`, `src/jssp_research/native/{__init__,bridge}.py`, `pyproject.toml` `[tool.maturin]` | **Partially implemented (build metadata) / Placeholder (code)** | `[tool.maturin]` is configured in `pyproject.toml`, but all 6 files in `crates/jssp-python/` and both files in `src/jssp_research/native/` are 0 bytes |
| 15 | **Experiment configuration & execution** | `configs/**/*.toml` (29), `src/jssp_research/{cli.py,config/*,experiments/*,benchmarks/*}` (19), `scripts/*.py` (6) | **Placeholder** | All 29 TOMLs, 19 Python package files, and 6 scripts are 0 bytes |
| 16 | **Result recording & reproducibility** | `crates/jssp-core/src/{rng,telemetry}/*` (5), `src/jssp_research/{results,analysis,reporting,utils}/*` (18) | **Placeholder** | All 5 Rust files, 18 Python files, and 4 reproducibility/schema tests are 0 bytes |

---

## 3. Existing Tests & Validation Command Results

### 3.1 Test & Fixture Inventory
All test and fixture files are currently **0-byte placeholders**:
- Rust tests (`5` files in `crates/jssp-core/tests/`): `decoder_correctness.rs`, `feasibility.rs`, `neighbourhood_round_trip.rs`, `obr_properties.rs`, `reproducibility.rs`.
- Python unit tests (`4` files in `tests/python/`): `test_config_validation.py`, `test_experiment_matrix.py`, `test_result_schema.py`, `test_seed_derivation.py`.
- Integration tests (`3` files in `tests/integration/`): `test_end_to_end_small_instance.py`, `test_python_rust_bridge.py`, `test_reproducible_run.py`.
- Test fixtures (`3` files in `tests/fixtures/`): `tiny_jssp.txt`, `medium_jssp.txt`, `expected_schedules.json`.

### 3.2 Validation Commands Executed & Outcomes

| Command | Exit Code | Observed Output / Blocker |
|---|---:|---|
| `cargo --version` / `rustc --version` / `cargo fmt --all -- --check` | `1` | `error: could not parse override file: '/home/abdelhalimyasser/Downloads/jssp-research/rust-toolchain.toml': empty toolchain override file detected.` |
| `rustup run stable cargo fmt --all -- --check` | `101` | `error: failed to parse manifest at /home/abdelhalimyasser/Downloads/jssp-research/Cargo.toml` (`manifest is missing either a [package] or a [workspace]`) |
| `rustup run stable cargo test --workspace --locked` | `101` | Same `Cargo.toml` manifest parse failure (root `Cargo.toml` is 0 bytes) |
| `rustup run stable cargo clippy --workspace --all-targets --all-features --locked` | `101` | Same `Cargo.toml` manifest parse failure (root `Cargo.toml` is 0 bytes) |
| `uv lock --check` | `1` | `error: The lockfile at uv.lock needs to be updated, but --check was provided. hint: To update the lockfile, run uv lock.` |
| `uv run --locked pytest` | `1` | Blocked by `uv.lock` out-of-sync with `pyproject.toml`; additionally, `pytest` is absent from `.venv/bin/` |
| `uv run --locked ruff check .` | `1` | Blocked by `uv.lock` out-of-sync with `pyproject.toml`; additionally, `ruff` is absent from `.venv/bin/` |
| `.venv/bin/ruff check .` / `.venv/bin/pytest` | `127` | `No such file or directory` — dev dependencies were never synced into `.venv` |
| `.venv/bin/python -c "import sys, numpy, pandas, scipy, matplotlib, pydantic; ..."` | `0` | Runtime imports succeed (`Python 3.15.0rc2`, `numpy 2.5.5`, `pandas 3.1.1`, `scipy 1.18.1`, `matplotlib 3.11.2`, `pydantic 2.14.4`) |

---

## 4. Development Environment Validation

| Component | Expected / Configured | Actual Installed / Workspace State | Status / Finding |
|---|---|---|---|
| **Python** | `3.15.2` (prompt & `AGENTS.md`); `>=3.15,<3.16` (`pyproject.toml`); `3.15` (`.python-version`) | `Python 3.15.0rc2` in `.venv` (`cpython-3.15.2` shown `<download available>` in `uv python list`) | **Mismatch:** `.venv` uses `3.15.0rc2` instead of `3.15.2`; `.python-version` lacks patch pin `3.15.2` |
| **uv** | `>=0.13.9,<0.14` (`pyproject.toml`) | `uv 0.13.9 (x86_64-unknown-linux-gnu)` | **OK** (matches constraint; `uv.lock` stale vs `pyproject.toml`) |
| **NumPy** | `>=2.5.5` (`pyproject.toml`) | `2.5.5` (in `uv.lock` & `.venv`) | **OK** |
| **pandas** | `>=3.1.1` (`pyproject.toml`) | `3.1.1` (in `uv.lock` & `.venv`) | **OK** |
| **SciPy** | `>=1.18.1` (`pyproject.toml`) | `1.18.1` (in `uv.lock` & `.venv`) | **OK** |
| **Matplotlib** | `>=3.11.2` (`pyproject.toml`) | `3.11.2` (in `uv.lock` & `.venv`) | **OK** |
| **Pydantic** | `>=2.14.4` (`pyproject.toml`) | `2.14.4` (in `uv.lock` & `.venv`) | **OK** |
| **Rust** | `1.100.0` (M0 prompt); `rust-toolchain.toml` (0 bytes) | `1.98.1` (`cargo 1.98.1`, `rustc 1.98.1`, `stable-x86_64-unknown-linux-gnu`) | **Blocked / Mismatch:** 0-byte `rust-toolchain.toml` breaks `cargo`/`rustc`; installed toolchain is `1.98.1`, not `1.100.0` |
| **PyO3** | `bindings = "pyo3"` (`pyproject.toml`) | Not specified (`crates/jssp-python/Cargo.toml` & `Cargo.lock` are 0 bytes) | **Missing** |
| **Maturin** | `>=1.16,<2.0` (`[build-system]` & `dev` group in `pyproject.toml`) | Not in `uv.lock`; not in `.venv/bin/` | **Missing in lock/venv** |
| **Ruff** | `>=0.17.6,<0.18` (`dev` group in `pyproject.toml`) | Not in `uv.lock`; not in `.venv/bin/` | **Missing in lock/venv** |
| **pytest / pytest-cov / hypothesis** | `dev` group in `pyproject.toml` | Not in `uv.lock`; not in `.venv/bin/` | **Missing in lock/venv** |

---

## 5. Research Configuration & Repository Discrepancies

1. **Planned 67 Configurations vs. Empty `configs/`:** `AGENTS.md` and `README.md` define 24 Phase A + 32 (or 28) Phase B + 11 Phase C = 67 (or 63) configurations, while all 29 TOML files in `configs/` are 0 bytes. Matrix integrity cannot yet be programmatically tested.
2. **`pyproject.toml` vs. `uv.lock` Drift:** `pyproject.toml` has uncommitted changes adding `[build-system]`, `[project.scripts]`, `[dependency-groups].dev`, `[tool.maturin]`, `[tool.ruff]`, and `[tool.pytest.ini_options]`, whereas `uv.lock` still holds `requires-python = ">=3.12"` and only the 19 runtime/transitive packages. Consequently `uv lock --check` and `--locked` commands fail, and dev tools (`ruff`, `pytest`, `maturin`, `hypothesis`) are absent from `.venv`.
3. **Python Patch Pin Discrepancy:** `.python-version` specifies `3.15` and `.venv` runs `3.15.0rc2`, whereas `AGENTS.md` §8 and M0 specify `3.15.2` (available via `uv python install 3.15.2`).
4. **Rust Toolchain Blocker & Version Discrepancy:** Empty 0-byte `rust-toolchain.toml` causes `rustup`/Cargo shim invocations to abort immediately. Furthermore, the host has `stable` (`1.98.1`), whereas M0 mentions `1.100.0`. Both `rust-toolchain.toml` and `Cargo.toml` workspace manifests need explicit alignment.
5. **PyO3 / Native Module Consistency:** `pyproject.toml` configures `module-name = "jssp_research._native"` and `manifest-path = "crates/jssp-python/Cargo.toml"`, which is internally consistent in Python config, but `crates/jssp-python/Cargo.toml` and `src/jssp_research/native/` are 0 bytes, and `crates/jssp-python` cannot yet build (`maturin` missing, `Cargo.toml` empty).
6. **US vs. UK Spelling Inconsistency Across Files:**
   - Rust source directory uses US spelling (`crates/jssp-core/src/neighborhoods/`), while Rust test (`crates/jssp-core/tests/neighbourhood_round_trip.rs`) and config (`configs/phase_b/neighbourhoods.toml`) use UK spelling. Must be preserved or harmonized carefully so imports and filenames remain unambiguous.
7. **License Metadata Mismatch:** `pyproject.toml` has `# license = "MIT"` and `# license-files = ["LICEN[CS]E*"]` commented out, while `LICENSE` and `NOTICE` are 0-byte placeholders.
8. **Dirty Working Tree:** `AGENTS.md`, `README.md`, `pyproject.toml`, and `uv.lock` have uncommitted changes relative to initial scaffold commit `36564e2aacd312218b56c7413181c55a10ff14a8`.

---

## 6. Missing Components Inventory

- **Tooling & Workspace Configs:** Root `Cargo.toml`, `crates/jssp-core/Cargo.toml`, `crates/jssp-python/Cargo.toml`, `Cargo.lock`, valid `rust-toolchain.toml`, `rustfmt.toml`, `.gitignore`, `LICENSE`, `NOTICE`, `CITATION.cff`, synced `uv.lock`, and pinned Python `3.15.2`.
- **Algorithm & Architecture Docs (pre-existing placeholders):** `docs/algorithms/{obr_decoder,qmc_mapping,n5_n6_n7_n8_definitions,mh_vs_sa,dual_view}.md`, `docs/architecture/{overview,data_flow,python_rust_boundary}.md`, `docs/benchmarks/{catalog,benchmark_provenance}.md`, `docs/decisions/ADR-0001-python-rust-workspace.md`, `docs/experiments/{fairness_protocol,phase_a,phase_b,phase_c}.md`.
- **Rust Core & Bindings:** All 70 source files + 5 test files in `crates/jssp-core/`, all 5 source files in `crates/jssp-python/`, and all 4 Rust benches in `benches/rust/`.
- **TOML Configuration Files:** All 29 TOML files in `configs/`.
- **Python Orchestration & Scripts:** All 39 files in `src/jssp_research/` and 6 scripts in `scripts/`.
- **Tests, Fixtures & Benchmark Data:** All 10 files in `tests/` and all benchmark raw/metadata/bounds/split artifacts in `data/`.

---

## 7. Technical & Scientific Risks

| ID | Risk | Severity | Impact & Mitigation |
|---|---|---|---|
| **R1** | **Broken Rust/Python Tooling Baseline** | **High** | 0-byte `rust-toolchain.toml` & `Cargo.toml` prevent compiling/linting; stale `uv.lock` blocks `pytest`/`ruff`/`maturin`. Fix workspace manifests and lockfiles first in M1. |
| **R2** | **Ambiguous N5–N8 Literature Definitions** | **High** | Mixing incompatible published variants or using naive OBR swaps invalidates Phase B comparisons. Freeze primary citations and graph preconditions in `docs/algorithms/n5_n6_n7_n8_definitions.md` before coding. |
| **R3** | **OBR Adapter Round-Trip Distortion** | **High** | Some disjunctive-graph moves can silently alter unintended machine precedence upon re-encoding to OBR. Enforce mandatory topological round-trip assertion and explicit move rejection telemetry. |
| **R4** | **SA vs. Standalone MH Equivalence** | **Medium** | Implementing cooling or symmetric proposal acceptance without Hastings correction turns `B25`–`B28` into duplicate SA runs. Formalize fixed-target $\pi(S)$ and asymmetric $q(S' \mid S)$ in `docs/algorithms/mh_vs_sa.md`, or drop `B25`–`B28` (63 total). |
| **R5** | **Dual-View Budget Leakage or Unfair Capacity** | **High** | Counting pair IDs instead of total genotypes or undercounting Baldwinian/Lamarckian local-search evaluations invalidates Phase C claims. Enforce shared evaluation budget counters in Rust and unit tests. |
| **R6** | **Discrete QMC Mapping Degradation** | **Medium** | Sequential adaptive inverse-CDF selection over changing eligible job sets $R$ breaks continuous low-discrepancy guarantees. Measure diversity/duplicates directly rather than assuming theoretical bounds transfer. |
| **R7** | **Benchmark Leakage / Non-Rectangular Mixing** | **High** | Tuning SA/ILS/Dual-View on held-out instances or pooling Track 4/5 with Track 1 compromises scientific validity. Enforce frozen `data/splits/` and manifest checks. |

---

## 8. Recommended Implementation Order (Milestones M1–M7)

1. **M1 — Environment, Workspace & Algorithm Specification Freeze:**
   - Resolve toolchain versions (`rust-toolchain.toml` vs host `1.98.1` or upgrade to `1.100.0`; install/pin Python `3.15.2`; `uv lock && uv sync --all-groups`), populate root & crate `Cargo.toml` (with `pyo3`), `.gitignore`, `rustfmt.toml`, and lock `Cargo.lock`.
   - Resolve scientific decisions `D01`–`D07` in `docs/algorithms/*`, `docs/architecture/*`, `docs/decisions/*`, and `docs/experiments/*`.
2. **M2 — Rust Core Foundation (`jssp-core` Model, OBR, Decoder, Feasibility, Makespan, Telemetry, RNG) + Fixtures:**
   - Implement `model`, `representation`, `decoder`, `feasibility`, `objective`, `rng`, `telemetry`, `error`, plus tiny/medium fixtures (`tests/fixtures/`). Verify `cargo test` and property tests before any metaheuristic work.
3. **M3 — PyO3 Bridge & Python Config/Matrix/Runner Infrastructure:**
   - Implement `crates/jssp-python`, `src/jssp_research/{config,benchmarks,experiments,native,results,utils}` and populate all 29 TOML configs. Verify `maturin develop`, `test_config_validation.py`, `test_experiment_matrix.py`, and smoke integration tests.
4. **M4 — Phase A Implementation & Screening Pilot:**
   - Implement `initialization/` (`A01`–`A24`), QMC discrete mapping, diversity metrics, and pilot runs.
5. **M5 — Phase B Implementation (Critical Graph, N5–N8 Adapter, Local Search & Metaheuristics):**
   - Implement `graph/`, `neighborhoods/` with mandatory OBR round-trip verification, `local_search/`, and `metaheuristics/` (`B01`–`B32` or pruned `28`). Validate MH Hastings ratio, SA cooling, and ILS perturbation tests.
6. **M6 — Phase C Implementation (Classical GA, Memetic GAs & Dual-View):**
   - Implement `genetic/`, `memetic/` (`C01`–`C04`), and `dual_view/` (`C05`–`C11`) with strict budget/lineage/gate invariants and correctness tests.
7. **M7 — Benchmark Curation, Confirmatory Runs & Paper Output Pipeline:**
   - Fetch/checksum Tracks 1–3 (and optionally 4–5), freeze splits and confirmatory protocol, execute full confirmatory runs, and implement `analysis/`, `reporting/`, and `scripts/` to generate reproducible tables and figures.
