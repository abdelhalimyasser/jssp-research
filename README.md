# Classical Job-Shop Scheduling Research Framework

> A reproducible research framework for studying population initialization, neighborhood-based local search, metaheuristics, and memetic genetic algorithms for the **classical Job-Shop Scheduling Problem (JSSP)**.

**Primary objective:** minimize the makespan $C_{\max}$.

**Implementation:** Python for experiment orchestration and analysis; Rust for computationally intensive scheduling algorithms; PyO3/Maturin for the Python–Rust boundary.

---

## Table of Contents

- [Research Scope](#research-scope)
- [Research Questions](#research-questions)
- [Experimental Design](#experimental-design)
  - [Phase A — Initial Population Generation](#phase-a--initial-population-generation)
  - [Phase B — Local Search and Metaheuristics](#phase-b--local-search-and-metaheuristics)
  - [Phase C — Memetic Genetic Algorithms](#phase-c--memetic-genetic-algorithms)
- [Benchmark Strategy](#benchmark-strategy)
- [Fairness and Reproducibility](#fairness-and-reproducibility)
- [Software Architecture](#software-architecture)
- [Technology Stack](#technology-stack)
- [Repository Layout](#repository-layout)
- [Development Setup](#development-setup)
- [Testing and Validation](#testing-and-validation)
- [Experiment Workflow](#experiment-workflow)
- [Results and Reporting](#results-and-reporting)
- [Research Integrity](#research-integrity)
- [Project Completion Checklist](#project-completion-checklist)

---

## Research Scope

This project studies the **classical deterministic, non-preemptive JSSP** under the makespan-minimization objective.

Given a set of jobs and machines:

- Every operation has a specified machine and processing time.
- Every job has a fixed operation precedence order.
- A machine processes at most one operation at a time.
- An operation cannot be interrupted once processing starts.
- The objective is to minimize the completion time of the last operation:

$$
C_{\max}=\max_{o\in O} C_o
$$

The primary representation is **Operation-Based Representation (OBR)**. Each valid chromosome encodes the required number of occurrences of each job. A common, documented decoder maps OBR chromosomes to feasible schedules. All methods in a direct comparison must use the same representation, decoder semantics, feasibility rules, and makespan evaluator.

### In scope

- Standard rectangular classical JSSP instances.
- OBR-based schedule construction and evaluation.
- Random, Monte Carlo, and Quasi-Monte Carlo (QMC) population initialization.
- MWR-biased operation selection during schedule construction.
- N5, N6, N7, and N8 neighborhood structures, using source-verified definitions.
- Local search and classical metaheuristics.
- Genetic and memetic algorithms, including a proposed paired Dual-View design.
- Reproducible benchmarking, statistical analysis, and paper-ready reporting.

### Out of scope for the primary comparison

Flexible JSSP, unrelated machine environments, online arrivals, sequence-dependent setup times, stochastic processing times, and multi-objective scheduling are not part of the primary classical JSSP claim. Variable-route/re-entrant and industrial instances may be studied in separate extension tracks and must not be pooled with standard rectangular JSSP results without an explicit model definition.

No machine learning or model training is required by the current research plan. The adaptive Dual-View gate is intended to be a transparent, rule-based mechanism unless a separately approved research revision changes that scope.

---

## Research Questions

The experiments are designed to investigate the following questions.

1. **Initialization:** How do Random, Monte Carlo, Halton, Sobol, scrambled QMC, and MWR-biased variants affect initial-schedule quality and population diversity?
2. **QMC mapping:** How does the discrete mapping from QMC coordinates to eligible-job choices affect the resulting OBR population?
3. **Neighborhoods:** How do N5, N6, N7, and N8 differ in move availability, feasibility, evaluation cost, and solution improvement?
4. **Search strategy:** How do descent methods, Tabu Search, ILS, Simulated Annealing, and correctly specified Metropolis–Hastings-based search compare under a common computational budget?
5. **Memetic learning:** How do Lamarckian, Baldwinian, and Partial Lamarckian learning affect GA search quality, diversity, and computational cost?
6. **Dual-View proposal:** Does retaining paired Lamarckian and Baldwinian genotypes for the same individual, with a fixed or adaptive pair-level gate, provide measurable benefits over single-view and unpaired controls?
7. **Generalization and scalability:** How do selected methods behave across classical benchmark families, generated instance spaces, larger rectangular problems, and separately defined variable-route/industrial extensions?

These are research questions, not assumed outcomes. No method is presumed superior before evaluation.

---

## Experimental Design

The current design contains **67 planned configurations**:

| Phase | Purpose | Planned configurations |
|---|---|---:|
| A | Initial population generation | 24 |
| B | Local search and metaheuristics | 32 |
| C | Memetic genetic algorithms | 11 |
| **Total** | | **67** |

The count is a count of algorithm configurations, not the number of executions. The final number of runs also depends on the selected instances, independent seeds, and protocol. If standalone Metropolis–Hastings is found to be algorithmically redundant with the chosen Simulated Annealing implementation, its four neighborhood-specific configurations must be removed, reducing the planned total from 67 to 63. This decision must be documented before confirmatory experiments.

### Phase A — Initial Population Generation

Phase A studies how the initial population is constructed. It does not change the GA operators or the decoder.

| ID | Initializer | MWR exponent $\alpha$ / ratio |
|---|---|---:|
| A01 | Uniform Random | — |
| A02 | Deterministic MWR | deterministic rule |
| A03 | Monte Carlo + MWR bias | 1.0 |
| A04 | Halton | — |
| A05 | Scrambled Halton | — |
| A06 | Sobol | — |
| A07 | Scrambled Sobol | — |
| A08 | Halton + MWR bias | 1.0 |
| A09 | Scrambled Halton + MWR bias | 1.0 |
| A10 | Sobol + MWR bias | 1.0 |
| A11 | Scrambled Sobol + MWR bias | 1.0 |
| A12 | Mixed Halton/Sobol population | 50:50 |
| A13 | Mixed Halton/Sobol population | 25:75 |
| A14 | Mixed Halton/Sobol population | 75:25 |
| A15–A16 | Monte Carlo + MWR bias | 0.5, 2.0 |
| A17–A18 | Halton + MWR bias | 0.5, 2.0 |
| A19–A20 | Scrambled Halton + MWR bias | 0.5, 2.0 |
| A21–A22 | Sobol + MWR bias | 0.5, 2.0 |
| A23–A24 | Scrambled Sobol + MWR bias | 0.5, 2.0 |

For MWR-biased selection, the intended probability of selecting eligible job $j$ is:

$$
P(j)=\frac{W_j^{\alpha}}{\sum_{k\in R}W_k^{\alpha}}
$$

where $R$ is the set of currently eligible jobs and $W_j$ is the remaining processing workload of job $j$, as formally defined in the implementation specification. $\alpha=0$ gives uniform selection; larger positive values increase the bias toward jobs with more remaining work.

**Required implementation decisions:**

- Specify precisely how each QMC coordinate is mapped to a discrete choice among the currently eligible jobs (for example, sorted eligible jobs plus inverse-CDF selection).
- Specify how coordinates are consumed across chromosome positions and how dimensions are handled.
- Use valid, documented scrambling methods and independent scramble seeds for randomized replicates.
- Use appropriate Sobol sample-size practices, including power-of-two sample sizes where required by the chosen construction.
- Do not claim that low-discrepancy properties are automatically preserved after adaptive discrete mapping; measure the resulting schedule and population properties.
- Define deterministic MWR tie-breaking. A deterministic MWR run may generate duplicate chromosomes; report this honestly rather than introducing undocumented randomness.
- Mixed Halton/Sobol means allocating individuals in the population to the two generators at the stated ratios. It does not mean naively adding or interleaving raw sequence coordinates.

**Phase A evaluation:** assess initial best/mean/median makespan, population diversity, duplicate rate, feasible-schedule rate, and generation or initialization time. Then evaluate the selected initializers under an otherwise identical classical GA to measure their downstream effect.

### Phase B — Local Search and Metaheuristics

The neighborhood structures are:

$$
\mathcal{N}=\{N5,N6,N7,N8\}
$$

The planned matrix crosses each neighborhood with eight search methods:

| Method | N5 | N6 | N7 | N8 |
|---|:---:|:---:|:---:|:---:|
| Steepest Descent | B01 | B02 | B03 | B04 |
| First Improvement | B05 | B06 | B07 | B08 |
| Random Improvement | B09 | B10 | B11 | B12 |
| Tabu Search | B13 | B14 | B15 | B16 |
| Iterated Local Search (ILS) | B17 | B18 | B19 | B20 |
| Simulated Annealing (SA) | B21 | B22 | B23 | B24 |
| Metropolis–Hastings-based Search | B25 | B26 | B27 | B28 |
| Hybrid ILS + MH | B29 | B30 | B31 | B32 |

**Terminology and algorithm requirements:**

- N5–N8 are neighborhood definitions, not search strategies. Their names are not universally interchangeable across papers. Each implementation must cite and follow a precise source definition, including move-generation restrictions and feasibility conditions.
- Steepest Descent evaluates the applicable neighborhood and selects the best improving move within the remaining budget. First Improvement accepts the first improving move under a documented deterministic scan order. Random Improvement samples from improving moves using a specified RNG stream.
- ILS must specify its inner local search, perturbation operator, perturbation strength, acceptance rule, and stopping rule. The initial comparison should use the same inner local search across N5–N8, with Steepest Descent as the planned default unless the experiment protocol records another choice.
- SA uses a temperature schedule and a Metropolis-style acceptance probability. Its initial temperature, cooling rule, cooling factor, and stopping condition must be specified and tuned on development instances only.
- Standalone MH-based search must specify a target distribution $\pi$, proposal distribution $q$, and the correct Hastings acceptance ratio:

$$
\alpha(S,S')=\min\left(1,\frac{\pi(S')q(S\mid S')}{\pi(S)q(S'\mid S)}\right)
$$

  For a makespan-oriented target, one possible form is $\pi(S)\propto\exp(-\beta C_{\max}(S))$, but $\beta$ and the proposal must be explicitly defined. If the proposal is asymmetric, the Hastings correction must not be omitted.
- The Hybrid ILS + MH must state exactly where MH is applied. In this plan, the intended design is to use MH to accept/reject the ILS perturbation proposal before the embedded deterministic local-search descent. Do not claim detailed balance for the complete perturbation-plus-descent transition unless it is mathematically established.
- If standalone MH and SA use the same state transitions, target/temperature behavior, and acceptance rule in a way that makes them the same algorithm, remove the redundant MH row rather than presenting duplicate algorithms.

**Phase B evaluation:** use matched initial solutions and equal objective-evaluation budgets. Record best/mean/median makespan, runtime, number of evaluated moves, generated/feasible/improving move counts, acceptance rate where applicable, and anytime performance. Report both objective-evaluation-normalized and wall-clock results.

### Phase C — Memetic Genetic Algorithms

| ID | Method | Definition / purpose |
|---|---|---|
| C01 | Classical GA | Non-memetic reference baseline |
| C02 | Lamarckian GA (LGA) | Local-search improvement is written back to the genotype |
| C03 | Baldwinian GA (BGA) | Genotype is retained; fitness uses the locally improved phenotype |
| C04 | Partial Lamarckian GA (PLGA) | Planned baseline applies Lamarckian learning to 50% of eligible individuals; selection rule must be specified |
| C05 | Dual-View Fixed Gate, $p=0.5$ | Paired Lamarckian/Baldwinian views with fixed equal gate weight |
| C06 | Dual-View Adaptive Gate | Paired views with a rule-based adaptive gate using performance and diversity signals |
| C07 | Dual-View Fixed Gate, $p=0.25$ | Fixed-weight sensitivity experiment |
| C08 | Dual-View Fixed Gate, $p=0.75$ | Fixed-weight sensitivity experiment |
| C09 | Adaptive Gate without diversity signal | Ablation of the diversity input |
| C10 | Unpaired Dual-Population Control | Tests the value of pairing/identity rather than simply running two populations |
| C11 | Controlled Cross-View Migration | Exploratory ablation; migration rate and transfer semantics must be fixed |

#### Proposed Dual-View representation

A paired individual is represented conceptually as:

$$
I_i=(ID_i,G_i^L,G_i^B)
$$

where both views share an individual identity and originate from the same initial OBR chromosome:

- **Lamarckian view** $G_i^L$: local-search improvements are written into the genotype and can be inherited by descendants of that view.
- **Baldwinian view** $G_i^B$: the genotype is retained; fitness is evaluated using its locally improved phenotype.
- In the base Dual-View design, crossover and mutation occur within the same view type. Cross-View Migration is isolated in C11 rather than silently included in the base method.
- Both views remain represented; the gate changes their influence on paired parent selection/reproductive allocation rather than deleting one view.

The gate must have a fully specified, reproducible rule. A possible pair-level score is:

$$
S_i(t)=p_t S_i^L+(1-p_t)S_i^B
$$

where $S_i^L$ and $S_i^B$ are normalized selection scores and $p_t\in[0,1]$. This formula is a design candidate, not an automatic implementation instruction: score normalization, selection semantics, bounds, update frequency, and the adaptive update equation must be specified and justified before confirmatory experiments.

The adaptive gate is rule-based; it does not require training or machine learning. The performance and diversity signals, normalization, update interval, and lower/upper bounds must be recorded in configuration files.

**Dual-View cost accounting:** both views and all local-search calls must count toward the computational budget. A paired individual must not be treated as having the cost of a single genotype if both views are evaluated or evolved.

---

## Benchmark Strategy

Benchmark families are organized into separate tracks. Results from different problem variants must not be merged into one aggregate score without a clearly justified common model and objective.

### Track 1 — Core Classical Rectangular JSSP

The historical 242-instance suite consists of:

| Family | Instances |
|---|---:|
| FT | 3 |
| LA | 40 |
| ABZ | 5 |
| ORB | 10 |
| SWV | 20 |
| YN | 4 |
| TA | 80 |
| DMU | 80 |
| **Total** | **242** |

Retain this suite for continuity with historical work. Do not assume every instance remains difficult for modern solvers. Report performance by family and difficulty/optimality status where available.

### Track 2 — Extended Rectangular Scale Benchmark

Use the 90 Large-TA instances associated with Da Col and Teppan as an extended scale set. They cover nine job-machine size strata, from 10×10 through 1000×1000, with ten instances per stratum.

For planning and reporting:

- The eight strata below 1000×1000 form the 80-instance rectangular extension.
- The ten 1000×1000 instances are reported separately as extreme stress tests.
- Do not require all 67 configurations to run on all million-operation stress cases. Use a separately declared shortlist and runtime/memory budget for extreme-scale evaluation.

### Track 3 — Synthetic Instance-Space Generalization

The Strassl–Musliu dataset contains 4,225 generated JSSP instances. Use a stratified subset of approximately 100–200 instances initially, covering different sizes, jobs-to-machines ratios, and processing-time distributions.

Create development and held-out splits before tuning. Record the selection procedure and dataset provenance. The full 4,225-instance set is not required for the primary factorial comparison.

### Track 4 — Variable-Route / Re-entrant Known-Optima Extension

The 24 Da Col–Teppan known-optima instances contain variable job routes and possible machine revisits. They are not standard fixed-length rectangular OBR instances. Use them only after the representation, decoder, feasibility checks, and neighborhoods explicitly support variable routes and revisits. Report them separately.

### Track 5 — Industrial Extension (Optional)

The 20 BEL industrial instances are an optional extension. The original study's objective differs from makespan minimization; if they are re-evaluated for $C_{\max}$, document the transformation and objective change. Do not compare the project's makespan results directly to the original paper's weighted completion-time results.

### Bounds and benchmark provenance

Use a maintained source such as JSPLib for current best-known bounds and instance metadata where applicable. Archive the exact downloaded benchmark files, source URL, retrieval date, checksum, and best-known-bound snapshot used in every confirmatory experiment. A best-known value is not necessarily a proven optimum; label optimality status accurately.

---

## Fairness and Reproducibility

Fairness requirements apply to all reported comparisons.

### Common problem definition

- Same JSSP instance data and operation identifiers.
- Same OBR validity rules and decoder semantics for directly compared methods.
- Same feasibility validator and makespan evaluator.
- Same objective definition and integer/time-unit conventions.
- Same stopping and budget-accounting conventions.

### Computational budgets

- Use objective-evaluation budgets as the primary normalized budget for stochastic/local-search comparisons.
- Record wall-clock time, CPU details, peak memory where practical, and total move-generation/evaluation counts.
- Count both Dual-View branches and all embedded local-search evaluations.
- Record initialization cost separately from search cost, as well as total end-to-end cost.
- Do not compare iteration counts as if they implied equal computational effort.

### Randomness

- Store the master seed and derive named independent substreams for initialization, selection, crossover, mutation, neighborhood sampling, perturbation, scrambling, and acceptance decisions.
- Use multiple independent runs for stochastic algorithms. The initial protocol target is 20 seeds per instance/configuration, subject to a documented pilot-based resource assessment and a frozen confirmatory protocol.
- Use matched instance/seed blocks when comparing methods where the random streams have a meaningful common interpretation.
- Deterministic methods must use documented tie-breaking and traversal order.
- Reproducibility means the same code, configuration, instance, and seed reproduce the same result on the declared software/hardware environment; do not promise bitwise reproducibility across all hardware or compiler configurations without verification.

### Experimental separation

Use staged evaluation:

1. **Correctness pilot:** tiny and medium instances for implementation checks.
2. **Development/screening:** stratified development instances for parameter tuning and candidate selection.
3. **Confirmatory evaluation:** held-out instances with frozen algorithm parameters and analysis protocol.
4. **Scalability/extension evaluation:** shortlisted methods on large, stress, variable-route, or industrial tracks.

Do not tune an algorithm on held-out confirmatory instances and then report those same instances as an unbiased final test.

### Parameters

Every algorithm parameter must be explicit in configuration files, validated, and included in run manifests. Parameter tuning budgets should be balanced and reported. Do not silently use hand-tuned settings for one algorithm and defaults for another.

---

## Software Architecture

The repository uses a Python–Rust split.

### Rust computational core

Rust is the source of truth for scheduling semantics and performance-critical operations:

- JSSP model and identifiers.
- OBR representation and chromosome validation.
- Schedule decoder and feasibility checks.
- Makespan and graph/critical-path computations.
- Neighborhood move generation and validation.
- Initializers and QMC mapping.
- Local search and metaheuristics.
- GA and memetic algorithms.
- Deterministic RNG stream handling and run telemetry.

### Python research orchestration

Python is responsible for:

- Loading and validating TOML experiment configurations.
- Building and checking the experiment matrix.
- Loading benchmark metadata and instance splits.
- Scheduling runs and managing seeds/budgets.
- Calling the Rust extension through a coarse-grained API.
- Persisting run manifests and results.
- Statistical analysis, plots, tables, and paper exports.

Avoid per-neighbor Python↔Rust calls. Submit a complete run or sufficiently large batch to the native engine and return structured results.

### Single source of truth

Do not implement a second, semantically different decoder, makespan evaluator, or neighborhood engine in Python. Python may provide lightweight validation and analysis helpers, but the authoritative scheduling implementation belongs in Rust.

---

## Technology Stack

| Component | Responsibility |
|---|---|
| Python 3.15.x | Orchestration, configuration, analysis, reporting |
| NumPy | Numerical arrays and analysis support |
| pandas | Result tables and data manipulation |
| SciPy | Statistical procedures |
| Matplotlib | Publication-oriented figures |
| Pydantic | Configuration/result validation where appropriate |
| Rust stable toolchain | Scheduling engine and computational core |
| PyO3 | Rust bindings for Python |
| Maturin | Building/installing the native Python extension |
| uv | Python environments, dependencies, and lockfile |
| Ruff | Python linting and formatting |
| pytest / Hypothesis | Python and property-based tests |
| Cargo test / Clippy / rustfmt | Rust tests and quality checks |

Pin exact dependency versions in the lockfiles used for each experiment release. Keep Python dependency declarations in `pyproject.toml` and Rust dependencies in the relevant `Cargo.toml` files. Do not add ML frameworks unless the research scope is formally revised.

---

## Repository Layout

The repository's current layout is organized around these responsibilities (the actual checked-out tree is authoritative if it differs):

```text
jssp-research/
├── AGENTS.md
├── benches/                 # Rust and end-to-end performance benchmarks
├── configs/                 # Defaults, phases, matrices, benchmarks, protocols
├── crates/
│   ├── jssp-core/           # Authoritative scheduling and algorithm engine
│   └── jssp-python/         # PyO3 extension crate
├── data/
│   ├── raw/                 # Original benchmark files (respect licenses)
│   ├── processed/           # Validated/normalized data
│   ├── metadata/            # Instance metadata and provenance
│   ├── bounds/              # Best-known bounds and snapshots
│   └── splits/              # Development/held-out split definitions
├── docs/
│   ├── algorithms/          # Formal algorithm definitions
│   ├── architecture/        # Design and Python/Rust boundary
│   ├── benchmarks/          # Catalog and provenance
│   ├── decisions/           # Architecture decision records
│   ├── experiments/         # Protocols and phase specifications
│   └── research/            # Consolidated research plan and structure
├── paper/                   # Manuscript, bibliography, tables, figures
├── reports/                 # Analysis outputs and phase reports
├── results/
│   ├── raw/                 # Immutable raw experiment outputs
│   ├── manifests/           # Run metadata and provenance
│   ├── summaries/           # Aggregated results
│   ├── logs/                # Execution logs
│   └── checkpoints/         # Restartable run state (not normal Git artifacts)
├── scripts/                # Experiment and benchmark utilities
├── src/jssp_research/       # Python orchestration package
└── tests/                   # Rust, Python, integration, and fixtures
```

Large datasets, raw result dumps, logs, and checkpoints should not be committed indiscriminately. Use checksums, manifests, an appropriate release/archive, or external storage for large artifacts. Keep small metadata, protocols, and paper outputs under version control.

---

## Development Setup

The repository's `pyproject.toml`, `.python-version`, Cargo manifests, and lockfiles are authoritative. The following is the intended setup flow; adjust only when the checked-in configuration requires it.

### 1. Install the selected Python toolchain

```bash
uv python install 3.15
```

Use the exact patch release declared by `.python-version` when one is pinned.

### 2. Synchronize Python dependencies

```bash
uv lock
uv sync --all-groups
```

Commit `uv.lock` after reviewing dependency changes.

### 3. Build/install the native extension

```bash
uv run maturin develop --manifest-path crates/jssp-python/Cargo.toml
```

The module name in the PyO3 crate, Maturin configuration, and Python import path must agree. Rebuild after native Rust changes when using an editable development install.

### 4. Check the environment

```bash
uv run python --version
uv run python -c "import numpy, pandas, scipy, matplotlib; print('Python research environment OK')"
cargo --version
rustc --version
```

If an optional dependency group is not installed, install it explicitly using the project's declared group rather than adding packages ad hoc.

---

## Testing and Validation

Run checks from the repository root.

### Rust

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace
```

### Python

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

### Required correctness properties

- OBR chromosome contains exactly the required number of occurrences of every job.
- Decoder never violates job precedence or machine exclusivity.
- Every valid decoded schedule has a finite, correctly computed makespan.
- Independent feasibility validation agrees with the decoder's feasibility status.
- Neighborhood moves preserve validity or are rejected with an explicit reason.
- Critical-path/critical-block calculations agree with small hand-checkable fixtures.
- Same seed/configuration/instance reproduces the same result in the supported deterministic environment.
- Rust and Python boundary returns correctly typed, versioned result data.
- Experiment matrix expansion produces the intended configuration IDs and counts.
- Budget counters include all relevant evaluations, including embedded local search and both Dual-View branches.

Do not begin large benchmark campaigns until the correctness and property-based tests pass.

---

## Experiment Workflow

1. **Inspect and validate inputs.** Verify benchmark format, dimensions, processing times, operation routes, checksums, and metadata.
2. **Run correctness fixtures.** Use tiny hand-verifiable instances before larger problems.
3. **Run a pilot.** Estimate runtime, memory, and evaluation budgets for each experiment family.
4. **Freeze development protocol.** Select development instances, parameter ranges, seeds, and stopping budgets.
5. **Run screening experiments.** Use development data only to identify promising configurations.
6. **Freeze confirmatory configurations.** Record exact commit, lockfiles, parameters, seeds, and test instances.
7. **Run confirmatory experiments.** Do not change parameters after inspecting held-out results.
8. **Run extension/scalability experiments.** Use shortlisted algorithms on larger or different-model instances and report them separately.
9. **Analyze and archive.** Preserve raw outputs, manifests, checksums, scripts, statistical tests, and generated tables/figures.

The experiment runner should support resuming interrupted work without duplicating completed runs. Every run should have a unique identifier and an immutable manifest.

---

## Results and Reporting

At minimum, report:

- Best, mean, median, and standard deviation of makespan across runs.
- Relative percentage deviation (RPD) from the selected best-known reference, with the reference snapshot identified.
- Feasible-schedule rate and, where relevant, success rate against a stated target.
- Runtime and objective-evaluation counts.
- Anytime curves or quality-versus-budget summaries for methods where this is informative.
- Population diversity and duplicate rate for GA/initialization experiments.
- Move-generation, feasibility, improvement, and acceptance statistics for neighborhood/metaheuristic experiments.
- Dual-View gate trajectory, view contribution, per-view local-search cost, and migration statistics where applicable.
- Statistical comparisons with the test family, pairing/blocking design, multiple-comparison correction, and effect sizes documented.

Use instance-level results and family-stratified summaries. Avoid relying only on a grand mean that allows easy instances to dominate the conclusions. Distinguish proven optimum, best-known upper bound, and lower bound.

Paper-ready outputs must be generated from stored raw results through version-controlled scripts. Do not manually alter reported tables or figures without preserving the transformation and source data.

---

## Research Integrity

- Treat novelty as a hypothesis to verify, not a label to attach to an implementation.
- Cite the original definitions of N5–N8 and implement the published move sets faithfully.
- Acknowledge prior uses of Halton/Sobol and adaptive Lamarckian/Baldwinian learning.
- Position the proposed Dual-View contribution specifically around persistent paired views and the precisely defined pair-level gate, if the literature review confirms that distinction.
- Separate the effect of initialization, local search, metaheuristic acceptance, and memetic inheritance through controlled comparisons and ablations.
- Report negative and inconclusive results, not only the best outcomes.
- Do not compare unlike objectives or problem variants as if they were the same benchmark.
- Preserve source citations, data provenance, software versions, and experiment manifests.

---

## Project Completion Checklist

Use this checklist as the living project tracker. Check an item only when its implementation or evidence has been verified; the existence of a source file alone does not mean the corresponding algorithm or test is complete.

### 1. Repository and research specification

- [x] Review the current repository tree and reconcile it with the intended architecture.
- [x] Finalize `docs/research/RESEARCH_PLAN_V1.md` from the approved experimental plan.
- [x] Finalize `docs/research/PROJECT_STRUCTURE_V1.md` from the actual repository structure.
- [ ] Record unresolved research decisions and their resolution in `docs/decisions/`.
- [x] Verify all Phase A/B/C configuration IDs and matrix counts.
- [x] Freeze the primary research questions, hypotheses, and scope.

### 2. Problem model, OBR, and decoder

- [x] Finalize the classical JSSP instance schema and input validation.
- [x] Finalize OBR chromosome validity rules and occurrence counting.
- [x] Implement and validate the OBR decoder.
- [x] Independently verify job precedence and machine exclusivity.
- [x] Validate makespan calculations against hand-computed fixtures.
- [ ] Validate critical paths and critical blocks on small instances.
- [x] Document decoder tie-breaking and deterministic behavior.

### 3. Phase A — Initializers and QMC

- [ ] Implement Uniform Random initialization.
- [ ] Implement deterministic MWR initialization and tie-breaking.
- [ ] Implement Monte Carlo + MWR for $\alpha=0.5,1,2$.
- [ ] Implement Halton and scrambled Halton generators.
- [ ] Implement Sobol and scrambled Sobol generators.
- [ ] Implement and test the QMC-to-discrete OBR mapping.
- [ ] Implement QMC + MWR configurations for all specified exponents.
- [ ] Implement mixed Halton/Sobol populations at 25:75, 50:50, and 75:25.
- [ ] Verify sample-size, dimension, scrambling, and seed handling.
- [ ] Measure initial makespan, diversity, duplicate rate, and construction cost.
- [ ] Evaluate selected initializers under the same frozen GA.

### 4. Phase B — Neighborhoods and search methods

- [ ] Collect and verify primary references defining N5, N6, N7, and N8.
- [ ] Document exact move definitions, preconditions, and boundary cases.
- [ ] Implement N5 and validate generated moves.
- [ ] Implement N6 and validate generated moves.
- [ ] Implement N7 and validate generated moves.
- [ ] Implement N8 and validate generated moves.
- [ ] Implement Steepest Descent.
- [ ] Implement First Improvement with a documented scan order.
- [ ] Implement Random Improvement with explicit RNG handling.
- [ ] Implement Tabu Search and document tenure/aspiration rules.
- [ ] Implement ILS with fixed perturbation and acceptance definitions.
- [ ] Implement Simulated Annealing and document its cooling schedule.
- [ ] Specify and implement a mathematically correct MH target/proposal/acceptance rule.
- [ ] Verify whether standalone MH is distinct from the chosen SA implementation.
- [ ] Implement the explicitly defined ILS + MH hybrid, if retained.
- [ ] Track generated, feasible, improving, and evaluated moves.
- [ ] Compare all retained methods under matched starts and equal budgets.

### 5. Phase C — GA and memetic algorithms

- [ ] Validate the classical GA baseline and freeze its parameters.
- [ ] Implement Lamarckian learning and verify genotype write-back.
- [ ] Implement Baldwinian learning and verify genotype preservation/phenotype fitness.
- [ ] Implement Partial Lamarckian learning and freeze its selection proportion/rule.
- [ ] Specify the Dual-View individual and shared identity semantics.
- [ ] Specify Fixed Gate selection/reproductive semantics and tested weights.
- [ ] Specify the Adaptive Gate update equation, signals, normalization, and bounds.
- [ ] Implement Dual-View Fixed Gate.
- [ ] Implement Dual-View Adaptive Gate.
- [ ] Implement the adaptive-without-diversity ablation.
- [ ] Implement the unpaired dual-population control.
- [ ] Implement controlled cross-view migration only as a separate ablation.
- [ ] Verify within-view reproduction and prevent unintended cross-view crossover.
- [ ] Account for the full computational cost of both views.
- [ ] Compare GA variants with identical baseline settings and budgets.

### 6. Benchmarks and data provenance

- [ ] Validate and checksum the 242 historical classical instances.
- [ ] Validate the 80 non-stress Large-TA instances.
- [ ] Validate and separately catalog the ten 1000×1000 stress instances.
- [ ] Create a stratified 100–200 instance Strassl–Musliu sample.
- [ ] Freeze development and held-out splits before tuning.
- [ ] Validate the 24 variable-route known-optima instances before using them.
- [ ] Decide whether to include the 20 BEL industrial instances and document the objective conversion.
- [ ] Archive benchmark source URLs, licenses, retrieval dates, and checksums.
- [ ] Archive the best-known-bound snapshot and optimality labels.

### 7. Fairness, experiments, and reproducibility

- [ ] Define and validate equal objective-evaluation budgets.
- [ ] Define runtime measurement and hardware reporting procedures.
- [x] Define master seeds and named independent RNG substreams.
- [x] Define the number of runs and freeze the confirmatory seed list.
- [ ] Ensure all algorithms use the same starts where the comparison requires paired starts.
- [ ] Ensure all embedded Local Search costs are counted.
- [x] Validate experiment manifests and unique run identifiers.
- [ ] Validate restart/resume behavior without duplicating completed runs.
- [ ] Complete correctness pilots before large-scale experiments.
- [ ] Tune only on development instances.
- [ ] Freeze parameters before confirmatory held-out runs.

### 8. Statistical analysis and paper outputs

- [ ] Implement makespan, RPD, runtime, evaluation, and diversity metrics.
- [ ] Implement descriptive statistics and confidence intervals.
- [ ] Select and document appropriate paired/non-parametric statistical tests.
- [ ] Apply multiple-comparison correction where required.
- [ ] Report effect sizes and instance-family breakdowns.
- [ ] Generate anytime and scalability plots where applicable.
- [ ] Generate Phase A, B, and C result tables from raw data.
- [ ] Complete QMC, neighborhood, and Dual-View ablation analyses.
- [ ] Archive scripts and exact inputs for every paper figure/table.
- [ ] Verify all literature claims and bibliography metadata.
- [ ] Draft Methods, Experimental Setup, Results, Threats to Validity, and Limitations.
- [ ] Re-run the final analysis from immutable raw results and manifests.

### 9. Release and archival

- [ ] Ensure all tests and quality checks pass on a clean checkout.
- [ ] Record Git commit, Python lockfile, Cargo lockfile, compiler, and hardware details.
- [ ] Remove secrets, temporary files, and unlicensed data from the release.
- [ ] Archive final configurations, manifests, raw results, and analysis scripts.
- [ ] Tag the research release and prepare reproducibility instructions.

---

## Status

This README describes the **planned research and software workflow**. It is not evidence that every listed algorithm, experiment, or validation step has already been completed. The checklist at the end is the authoritative progress tracker and should be updated only after verification.
