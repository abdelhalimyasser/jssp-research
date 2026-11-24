# AGENTS.md — Classical JSSP Research Framework

**Project:** `jssp-research`
**Python distribution:** `jssp-research`
**Python import package:** `jssp_research`
**Rust workspace:** inspect the root `Cargo.toml` and preserve its actual workspace/package names.
**Primary objective:** deterministic classical Job-Shop Scheduling Problem (JSSP), minimizing makespan.
**Working research baseline:** this file plus the checked-in `configs/` and `docs/`; the formal `docs/research/` documents are to be created from repository evidence during documentation bootstrap.

This file is the repository-wide instruction set for every coding agent and contributor. Read it before changing code.

**Important repository-state rule:** `docs/research/RESEARCH_PLAN_V1.md` and `docs/research/PROJECT_STRUCTURE_V1.md` are not assumed to exist in the current repository. Do not block, fail, or ask the owner for these files. Do not refer to them as existing documents until they have actually been created and committed.

The current repository already has (verify each path before relying on it):

- `docs/algorithms/` — algorithm definitions and unresolved technical specifications.
- `docs/architecture/` — architecture and Python/Rust boundary.
- `docs/benchmarks/` — benchmark catalog and provenance.
- `docs/experiments/` — Phase A/B/C and fairness protocols.
- `docs/decisions/` — architecture decision records.
- `configs/` — defaults, phase settings, benchmark definitions, protocols, and experiment matrices.
- `crates/`, `src/`, `tests/`, and `benches/` — actual implementation and validation code.

**Required first action for a new agent session:** inspect the real repository state. Read this file, `README.md`, the root `Cargo.toml`, relevant crate manifests, `pyproject.toml`, toolchain/version files, lockfiles, the relevant `configs/` files, and the existing documentation/code/tests for the task. Use `find`/`tree` or equivalent to verify paths. A directory tree or an earlier plan is not proof that a file exists or that its contents match the plan.

**Documentation bootstrap:** after the initial audit, create `docs/research/` and add:

1. `docs/research/RESEARCH_PLAN_V1.md` — consolidate the scientific scope, experiment IDs/counts, fairness protocol, benchmark tracks, and research claims from this file and the *actual checked-in* `configs/` and `docs/` files.
2. `docs/research/PROJECT_STRUCTURE_V1.md` — describe the *actual checked-in* project architecture and module boundaries based on the real repository, not an imagined future tree.

These are documents the agent must create as an explicit, reviewable documentation task; they are not prerequisites and must not be cited as existing before creation. Perform the bootstrap in this order:

1. Produce a repository inventory and identify which expected files are present, absent, empty, or inconsistent.
2. Compare the actual TOML matrices/configurations with the experiment counts in this file. Treat checked-in configuration as implementation state, not automatic proof that the scientific protocol is correct.
3. Create `docs/research/RESEARCH_PLAN_V1.md` and `docs/research/PROJECT_STRUCTURE_V1.md` from verified repository evidence and this working research baseline.
4. Record all unresolved decisions and discrepancies in the new research plan. Do not implement ambiguous algorithms merely to complete a directory tree.
5. Only then continue with the specific implementation task, unless the owner explicitly asks for a different sequence.

If the current configs/docs disagree with this file, do not silently overwrite either side. Produce a discrepancy list, identify which items are implementation state versus intended research protocol, and request/record an explicit decision for scientific changes. Do not invent missing algorithm definitions, benchmark facts, or experiment parameters to make the documents look complete.

Until those two documents have been created and reviewed, this `AGENTS.md` is the repository-wide research and engineering baseline. Once they exist, keep them consistent with this file and the checked-in configs. A task request must not silently alter a frozen confirmatory protocol; explain conflicts and preserve the approved protocol unless the project owner explicitly approves a versioned change. Never rewrite scientific plans as an incidental side effect of implementation.

---

## 1. Mission and scope

Implement a reproducible research framework for the **deterministic, classical, rectangular JSSP**, with makespan as the primary objective:

\[
\min C_{\max} = \min \max_j C_j.
\]

The core model has fixed job operation order, fixed operation-machine assignment, fixed processing times, non-preemptive operations, and machine capacity one. Release dates, sequence-dependent setup times, flexible job shops, stochastic processing times, due-date objectives, and reentrant/non-rectangular routing are outside the primary core comparison. Such extensions must be kept in separately labelled tracks and must never be silently pooled with rectangular classical JSSP results.

This is a classical algorithms and experimental-computing project. **Do not introduce machine learning, model training, neural networks, learned policies, or online training** unless the owner explicitly changes the research scope. Rule-based adaptation is allowed when fully specified and reproducible.

## 2. Authoritative research plan

The current planned design has:

- **Phase A — Initial population generation: 24 configurations.** Random, deterministic MWR, Monte Carlo+MWR, Halton, Scrambled Halton, Sobol, Scrambled Sobol, QMC+MWR, mixed Halton/Sobol population ratios, and MWR-bias sensitivity settings.
- **Phase B — Local search/metaheuristics: 32 configurations.** Four neighbourhoods (N5–N8) crossed with Steepest Descent, First Improvement, Random Improvement, Tabu Search, ILS, Simulated Annealing, fixed-target MH search, and ILS+MH. If standalone MH is not mathematically distinct from SA in the final implementation, remove those four cells and revise the total to 63.
- **Phase C — Memetic GA: 11 configurations.** Classical GA, LGA, BGA, PLGA, Dual-View Fixed/Adaptive gates, fixed-gate settings, adaptive ablation, unpaired control, and exploratory cross-view migration.
- **Current planned total: 67 configurations**, or 63 if the standalone MH cells are proven redundant. This is the number of configurations, not the number of runs.

Do not create the full Cartesian product `24 × 32 × 11`. The phases answer different research questions. Use development/screening instances to select fixed components before confirmatory evaluation. Never tune on the held-out test instances.

### 2.1 Phase A invariants

- Use the same OBR representation, population size, decoder, and downstream GA settings across initializers.
- MWR-biased sequential construction uses:

  \[
  P(j)=\frac{W_j^\alpha}{\sum_{k\in R}W_k^\alpha},
  \]

  where `R` is the set of jobs with remaining operations and `W_j` is the total remaining processing time of job `j`.
- Primary bias strength is `alpha = 1.0`; sensitivity values are `0.5` and `2.0`.
- `alpha = 0` is uniform choice and is a control, not an additional MWR variant.
- Deterministic MWR must document its tie-break rule. Do not add hidden randomness to break ties. If it creates duplicate schedules, report that honestly as a low-diversity deterministic control.
- Mixed QMC means assigning declared proportions of the population to separate QMC generators. Do not average or concatenate raw Halton/Sobol coordinates without a defined mathematical mapping.
- QMC coordinates must be mapped to discrete eligible-job choices through a documented sequential mapping (for example, inverse-CDF mapping). Do not claim that the original continuous low-discrepancy guarantees automatically survive this adaptive discrete mapping.
- Respect the actual maximum dimension and sample-size requirements of the selected Sobol implementation. Document scrambling method and independent scramble seeds.

### 2.2 Phase B invariants

- N5, N6, N7, and N8 are **schedule/neighbourhood definitions**, not interchangeable names for arbitrary OBR swaps. Their definitions vary across literature. Implement each from a named, cited source, record the exact move rules, and test its feasibility conditions.
- Steepest Descent evaluates the declared feasible neighbourhood and chooses the best strict improvement. If the evaluation budget ends before the neighbourhood is exhausted, do not label the current state a local optimum.
- First Improvement uses a declared deterministic move ordering and accepts the first strict improvement.
- Random Improvement uses a separately seeded RNG stream and a declared sampling/selection rule.
- ILS must declare its perturbation kernel, perturbation strength, embedded local search, acceptance criterion, and stopping budget. The primary comparison uses the same embedded Steepest Descent configuration across ILS variants.
- Simulated Annealing uses a cooling temperature schedule.
- Standalone Metropolis–Hastings uses a fixed target distribution and explicit forward/reverse proposal probabilities. A representative target is `pi(S) ∝ exp(-beta * Cmax(S))`. The Hastings correction must use the actual proposal probabilities.
- Do not call SA and MH distinct methods if their transition kernels and acceptance rules are implemented identically. If the project cannot define a meaningful fixed-target MH method distinct from cooling SA, remove the four standalone MH cells and report 63 configurations.
- The ILS+MH hybrid is defined in the research plan: MH filters the raw perturbation proposal before the deterministic local search. Calculate the Hastings ratio for that perturbation kernel. Do not claim that the entire post-local-search process is an MH chain satisfying detailed balance.
- Every metaheuristic must support matched objective-evaluation budgets and must report wall-clock time separately.

### 2.3 Phase C invariants

- All GA variants share the same OBR-valid selection/crossover/mutation/elitism configuration unless a declared ablation specifically changes one factor.
- LGA writes the local-search-improved phenotype back into the genotype.
- BGA preserves the original genotype while using the locally improved phenotype for fitness evaluation.
- PLGA's primary setting is the declared 50% application rate. Do not silently change that rate between experiments.
- A paired Dual-View individual is represented conceptually as:

  `Individual = (individual_id, lamarckian_genotype, baldwinian_genotype)`.

- The two views retain a shared identity but have separate genotype lineages. In the base design, perform crossover within each view; do not perform cross-view crossover.
- The Fixed Gate uses a constant `p`. The Adaptive Gate is a deterministic, rule-based controller (not ML) using the documented performance/diversity signals and a bounded update. Its complete equation, normalization, update window, and bounds must be written in `docs/algorithms/dual_view.md` before confirmatory runs.
- The pair-level selection score and the way reproductive slots are allocated must be precisely defined. Do not implement an arbitrary weighted score or interpret `p` differently in different modules.
- Account for both genotype slots and both views' local-search work when comparing Dual-View to single-view GA methods. Equalize total genotype capacity and total objective-evaluation budget, not merely the number of pair IDs.
- C10 (unpaired dual-population control) and C11 (cross-view migration) are controls/ablations. Do not describe migration as part of the base independent Dual-View mechanism.
- Treat Dual-View novelty as a research hypothesis. Do not claim first-ever novelty until systematic literature review is completed.

## 3. Shared model, OBR, and decoder

### 3.1 OBR contract

For `n` jobs, where job `j` has `m_j` operations, chromosome length is:

\[
L = \sum_{j=1}^{n} m_j.
\]

The chromosome contains job IDs, with job `j` appearing exactly `m_j` times. The `k`th occurrence of job ID `j` represents operation `k` of that job. Validate chromosome length, job-ID range, and each job's multiplicity before decoding.

Do not substitute machine-sequence encoding or a second chromosome representation in the core experiment matrix without an approved plan revision.

### 3.2 Deterministic serial insertion decoder

Use one canonical, documented deterministic serial schedule-generation decoder for all initializers and GA variants:

1. Scan OBR from left to right.
2. For the next occurrence of job `j`, identify its next unscheduled operation.
3. Find the earliest feasible start time satisfying job precedence and machine availability.
4. Use the declared insertion/earliest-feasible-gap policy consistently.
5. Resolve all equal-time ties with fixed documented rules (never unspecified hash/map iteration order).
6. Return operation start/completion times, machine sequences, schedule graph data, and makespan.

The same instance and OBR must always produce the same result. Decoder details must agree with `docs/algorithms/obr_decoder.md` and `configs/defaults/decoder.toml`.

### 3.3 Schedule-neighbourhood adapter

When a neighbourhood is defined on machine sequences, critical blocks, or a disjunctive graph:

1. Decode OBR to the canonical schedule/graph.
2. Apply the exact published move.
3. Validate precedence, machine exclusivity, and graph acyclicity.
4. Evaluate the candidate makespan exactly.
5. Encode a retained graph back to OBR using a stable topological ordering with declared tie-breaking.
6. Decode again and assert that the intended schedule/machine-order constraints are preserved.

Never implement N5–N8 as a generic chromosome swap merely because it is convenient. A move that cannot be represented back in OBR without changing its intended machine-order structure must be rejected and documented as unsupported by the adapter.

## 4. Architecture and language boundaries

- **Rust is the sole source of truth** for scheduling semantics and compute-heavy algorithms: JSSP model, OBR validation, decoder, feasibility, makespan, critical path/blocks, neighbourhoods, initializers, local search, metaheuristics, GA, and memetic algorithms.
- **Python owns orchestration and research analysis:** configuration loading/validation, experiment matrix construction, benchmark catalog/provenance/splits, run scheduling, result manifests, analysis, statistical tests, plots, tables, and paper exports.
- Do not reimplement the decoder, feasibility checks, makespan logic, or neighbourhood semantics in Python. Python may validate returned outputs independently for tests, but the production implementation remains in Rust.
- Keep Python↔Rust calls coarse-grained: submit one run or a batch of runs, not one FFI call per move/neighbour.
- Use PyO3 only at the boundary. Do not call Python callbacks from inner Rust search loops.
- Use explicit typed errors at the Rust boundary. Convert them into clear Python exceptions with enough context for a failed run manifest.
- Do not use NumPy arrays as the core internal schedule representation unless profiling and an explicit design decision justify them. Rust owns the main data structures; NumPy is for Python-side analysis/interop where useful.
- Do not add heavy dependencies without a concrete need. No scikit-learn, PyTorch, TensorFlow, notebook server, web framework, or database is required for the current core project.

## 5. Reproducibility and experiment protocol

Every run must persist a resolved, immutable manifest containing at least:

- experiment/configuration ID and phase;
- full resolved parameters (no hidden defaults);
- benchmark family, instance ID, instance checksum, and source/provenance;
- run ID, root seed, and named substream seeds;
- git commit hash and dirty-working-tree status;
- Python, uv, Rust, Cargo, PyO3, Maturin, NumPy, SciPy, pandas, and Matplotlib versions as applicable;
- OS, CPU model, logical/physical core count when available, RAM, build profile, and compiler flags;
- stopping condition, objective-evaluation budget, local-search budget, and wall-clock limit;
- start/end timestamps, elapsed time, makespan, lower/upper bound used, feasibility status, and all algorithm-specific metrics.

Use independent named RNG streams for initialization, QMC scrambling, crossover, mutation, perturbation, SA proposals, and MH proposals. A change in one stream must not silently alter the other streams.

Stochastic methods use multiple independent seeds. Use paired starting conditions where appropriate. The default planning target is 20 runs per stochastic configuration/instance during confirmatory work unless the frozen protocol specifies otherwise. Do not invent extra random restarts inside an algorithm without recording them.

Equal-budget rules:

- **Phase A:** same population size and same downstream GA evaluation budget; record initial population quality/diversity separately.
- **Phase B:** same initial schedules and same candidate/objective-evaluation budget; report wall time separately.
- **Phase C:** same GA parameters and same total schedule/local-search evaluation budget; split Dual-View local-search work across both views.

Do not compare stochastic heuristics solely on iteration count. Record both search effort and machine runtime. Store raw records separately from aggregate reports. Never overwrite raw results; create a new run/version.

Use development/screening instances for parameter tuning and selection. Freeze code commit, configs, seeds/protocol, and benchmark split before confirmatory runs. Any post-freeze change creates a new experiment version and must be disclosed.

## 6. Benchmark policy

Maintain separate benchmark tracks:

1. **Core historical rectangular JSSP:** FT, LA, ABZ, ORB, SWV, YN, TA, DMU (242 classic instances).
2. **Extended rectangular scale:** 80 non-stress Large-TA instances; 10 1000×1000 TAI stress instances reported separately.
3. **Synthetic generalization:** stratified sample (initially 100–200) from the 4,225 Strassl–Musliu generated instances. Preserve the original data split/provenance and use a held-out test subset.
4. **Variable-route/known-optima extension:** 24 Da Col–Teppan Known-Optima instances. Separate from rectangular results; routes may be non-rectangular/reentrant.
5. **Industrial extension:** 20 BEL instances only after confirming the instance format and explicitly recomputing the project's Cmax objective. Do not reuse reported weighted-sum-completion-time scores as makespan results.

Record source URL, license/usage terms, instance checksum, parsing notes, and date-stamped best-known bounds. Snapshot the JSPLib/BKS reference used for a paper run. Do not silently replace historical bounds with later values in an already frozen result set.

## 7. Coding and testing standards

- Keep modules cohesive and small; avoid giant algorithm files and cross-layer imports.
- Public Python APIs must have type annotations and docstrings. Rust public types/functions must have meaningful docs where they cross module boundaries.
- Prefer explicit enums/config structs over stringly typed algorithm switches.
- No hidden global RNG, global mutable algorithm state, or non-deterministic iteration-order dependence.
- No placeholder algorithms, fake benchmark results, or fabricated best-known values. Do not mark a phase complete while code is a stub.
- Do not optimize before correctness tests pass. Keep a simple exact/reference implementation for testing optimized incremental evaluation.
- Add or update tests with every semantic change.

Required correctness checks include:

- parsing and validation of tiny known JSSP fixtures;
- OBR multiplicity, bounds, and operation-occurrence properties;
- decoder determinism and earliest-feasible placement;
- precedence and machine-exclusivity checks;
- makespan independently recomputed from returned completion times;
- neighbourhood move feasibility and encode/decode round trips;
- same seed/config/instance reproduces the same result for deterministic modes;
- independent RNG streams are stable and do not interfere;
- MH forward/reverse proposal probability tests;
- SA cooling and acceptance boundary tests;
- ILS perturbation and acceptance tests;
- LGA/BGA/PLGA genotype/phenotype semantics;
- Dual-View identity, lineage separation, gate bounds, budget accounting, and paired-vs-unpaired tests;
- experiment-matrix counts and duplicate/missing configuration ID checks.

Performance benchmarks must be separate from correctness tests. Use release builds for performance measurements and record hardware/compiler details. Do not use GitHub-hosted CI timings as paper performance results.

## 8. Python environment and commands

Use `uv` for all Python environment and dependency operations. Do not use global `pip`, `pip install --user`, Conda, or a manually managed venv for this repository.

Expected project setup:

```bash
uv python install 3.15.2
uv python pin 3.15.2
uv lock
uv sync --all-groups
```

`uv sync` installs the project in editable mode through Maturin. After editing Rust sources, force a fresh native rebuild with `uv run maturin develop --manifest-path crates/jssp-python/Cargo.toml` before Python integration tests.

Typical validation:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace
```

For Rust performance tests, use release mode and the repository's documented benchmark protocol. Do not run a long benchmark matrix just to validate a code change; use a tiny smoke experiment.

Keep `uv.lock` committed. Update it deliberately after changing dependencies; do not delete it or regenerate it unnecessarily. Do not run dependency upgrades during a frozen confirmatory experiment.

## 9. Change-management rules for coding agents

Before editing:

1. Read this file.
2. Inspect the actual repository files relevant to the task, including existing docs/configs, tests, `pyproject.toml`, root and crate `Cargo.toml` files, toolchain declarations, and lockfiles. Do not assume research reference documents exist unless verified.
3. Identify the smallest coherent task and the files that need changes.
4. State scientific/algorithmic ambiguity before choosing a definition.
5. If the research-plan and project-structure documents do not yet exist, do not make them a blocker for code work. Create them only as the explicit documentation-bootstrap task described above, using repository evidence.

While editing:

- Preserve existing working code and user changes.
- Do not replace complete configuration files without reading and merging their current contents.
- Do not invent sources or silently redefine named algorithms.
- Do not change configuration IDs or experiment budgets casually.
- Update documentation and tests alongside implementation.

Before finishing:

1. Run the narrowest relevant tests, then the full validation suite when practical.
2. Report exact commands and pass/fail results.
3. List files changed and decisions made.
4. State any tests not run and why.
5. Do not claim a benchmark or algorithm is complete without actual execution evidence.

## 10. Research integrity

The project must distinguish established algorithms from proposed combinations. In particular:

- N8 is published; do not claim it as new.
- QMC use in scheduling has prior work; do not claim first use of Halton/Sobol.
- Adaptive Lamarckian/Baldwinian mechanisms have prior work; distinguish the proposed persistent paired Dual-View architecture from existing adaptive selection.
- The proposed contribution is a hypothesis until supported by systematic related-work review and ablation evidence.
- Report negative results, failed configurations, and computational overhead honestly.
- Never fabricate citations, BKS values, runtime values, or statistical significance.

---

## 11. Milestone Git checkpoint and push policy

- Each future completed milestone gets exactly one dedicated Git commit.
- Commit only after the milestone's validation commands pass; if validation fails, do not create a completion commit.
- Push each milestone commit to the configured GitHub branch.
- Use a descriptive milestone-specific commit message.
- Never force-push or rewrite published history.
- Never commit secrets, generated native binaries (`.so`, `.pyd`, `.dylib`), build directories (`/target/`, `/build/`, `/dist/`), virtual environments (`.venv/`), raw datasets without approval, or large experimental outputs.
- Include the commit SHA, branch name, and push status in every milestone completion report.
