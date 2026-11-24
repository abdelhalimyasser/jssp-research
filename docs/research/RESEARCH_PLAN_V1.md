# RESEARCH_PLAN_V1 — Classical JSSP Research Specification

**Status:** M0 Research Specification (Consolidated Baseline)
**Project:** `jssp-research` (`jssp_research` Python package / `crates/jssp-core` & `crates/jssp-python` Rust workspace)
**Primary Objective:** Minimize makespan ($C_{\max}$) in deterministic classical rectangular Job-Shop Scheduling Problems (JSSP)
**Primary Sources:** [`AGENTS.md`](file:///home/abdelhalimyasser/Downloads/jssp-research/AGENTS.md), [`README.md`](file:///home/abdelhalimyasser/Downloads/jssp-research/README.md)
**Repository Audit Note:** As of Milestone M0, all 29 TOML files in `configs/` and all 15 non-research documentation files in `docs/` (`docs/algorithms/`, `docs/architecture/`, `docs/benchmarks/`, `docs/decisions/`, `docs/experiments/`) exist on disk as **0-byte empty placeholders**. Accordingly, this plan records the authoritative scientific protocol from `AGENTS.md` and `README.md` while explicitly flagging checked-in config and algorithm specifications that remain unpopulated or unresolved.

---

## 1. Research Scope & Research Questions

### 1.1 In-Scope
- **Deterministic, non-preemptive, classical rectangular JSSP** minimizing makespan $C_{\max}$.
- **Operation-Based Representation (OBR)** as the sole chromosome encoding across the primary experimental matrix, paired with a single canonical deterministic serial insertion decoder.
- **Phase A — Population Initialization (24 configurations):** Uniform Random, Deterministic MWR, Monte Carlo + MWR, Halton, Scrambled Halton, Sobol, Scrambled Sobol, QMC + MWR combinations, mixed Halton/Sobol populations, and MWR-bias sensitivity ($\alpha \in \{0.5, 1.0, 2.0\}$).
- **Phase B — Neighborhoods, Local Search & Metaheuristics (32 planned configurations; 28 if standalone MH is redundant with SA):** Source-verified N5, N6, N7, and N8 schedule neighborhoods executed via a strict disjunctive-graph/OBR adapter and crossed with Steepest Descent, First Improvement, Random Improvement, Tabu Search, Iterated Local Search (ILS), Simulated Annealing (SA), standalone Metropolis–Hastings (MH), and Hybrid ILS + MH.
- **Phase C — Memetic Genetic Algorithms (11 configurations):** Classical GA, Lamarckian GA (LGA), Baldwinian GA (BGA), Partial Lamarckian GA (PLGA), and the proposed persistent paired **Dual-View** architecture (Fixed Gate, rule-based Adaptive Gate, sensitivity variants, diversity ablation, unpaired dual-population control, and controlled cross-view migration).
- **Reproducible Benchmarking & Statistical Analysis:** Staged evaluation (correctness pilot, development/screening, frozen confirmatory, scalability/extensions) with immutable manifests and non-parametric statistical testing.

### 1.2 Out-of-Scope for the Primary Comparison
- Flexible JSSP, unrelated parallel machines, release dates, sequence-dependent setup times, stochastic processing times, online/dynamic job arrivals, and due-date or multi-objective formulations.
- **No Machine Learning:** Neural networks, reinforcement learning, learned heuristic policies, and online training are strictly excluded from the current project scope. Any adaptive mechanism (such as the Dual-View Adaptive Gate in `C06`) must be transparent, deterministic, and rule-based.
- **Track Isolation:** Non-rectangular/re-entrant routing instances (Track 4: Da Col–Teppan Known-Optima) and alternative-objective industrial instances (Track 5: BEL) must never be silently pooled into rectangular classical JSSP aggregate metrics.

### 1.3 Core Research Questions (RQ1–RQ7)
1. **RQ1 (Initialization):** How do Uniform Random, Deterministic MWR, Monte Carlo + MWR, Halton, Sobol, scrambled QMC, and mixed QMC populations affect initial-schedule quality, population diversity, duplicate rate, and downstream GA performance?
2. **RQ2 (QMC Discrete Mapping):** How does sequential mapping from continuous QMC coordinates to discrete eligible-job choices affect the structural distribution and diversity of resulting OBR populations?
3. **RQ3 (Neighborhoods):** How do source-verified N5, N6, N7, and N8 neighborhoods differ in move availability, feasibility rate, adapter overhead, evaluation cost, and solution improvement?
4. **RQ4 (Search Strategies):** How do deterministic/randomized descent, Tabu Search, ILS, SA, and a mathematically valid fixed-target MH search compare under matched starts and equal objective-evaluation budgets?
5. **RQ5 (Memetic Learning):** How do Lamarckian, Baldwinian, and Partial Lamarckian inheritance rules alter GA convergence, genotype/phenotype diversity, and total search cost?
6. **RQ6 (Dual-View Hypothesis):** Does retaining persistent paired Lamarckian and Baldwinian genotypes sharing an individual identity, governed by a fixed or rule-based adaptive pair-level gate, outperform single-view memetic GAs (`C02`–`C04`) and an unpaired dual-population control (`C10`) under equalized genotype capacity and total evaluation budgets?
7. **RQ7 (Generalization & Scalability):** How do shortlisted configurations generalize across historical families, synthetic instance spaces (Strassl–Musliu), large rectangular scales (Large-TA), extreme $1000 \times 1000$ stress instances, and separately modeled extensions?

---

## 2. Formal Problem Definition & Objective Function

### 2.1 Classical Rectangular JSSP ($J_m \parallel C_{\max}$)
Let:
- $J = \{0, 1, \dots, n - 1\}$ be a set of $n$ independent jobs.
- $M = \{0, 1, \dots, m - 1\}$ be a set of $m$ machines.
- Each job $j \in J$ consists of a strictly ordered sequence of $m_j$ non-preemptive operations:
  $$O_{j,0} \prec O_{j,1} \prec \dots \prec O_{j, m_j - 1}$$
  In the **rectangular** classical JSSP, every job visits every machine exactly once ($m_j = m$ and $\bigcup_{k=0}^{m-1} \{\mu(O_{j,k})\} = M$). Total operations $L = \sum_{j=0}^{n-1} m_j = n \cdot m$.
- Each operation $o = O_{j,k}$ has a fixed assigned machine $\mu(o) \in M$ and a fixed deterministic integer processing time $p_o \in \mathbb{Z}_{\ge 0}$ (standard benchmarks strictly use $p_o \in \mathbb{Z}_{> 0}$; zero-duration handling must be explicitly verified during model specification).

### 2.2 Decision Variables & Feasibility Constraints
A schedule assigns a start time $S_o \ge 0$ and completion time $C_o = S_o + p_o$ to every operation $o \in O$, subject to:
1. **Non-preemption & Integrality:** $S_o \in \mathbb{Z}_{\ge 0}$ and $C_o = S_o + p_o$ for all $o \in O$.
2. **Job Precedence:** For every job $j \in J$ and operation index $k \in \{1, \dots, m_j - 1\}$:
   $$S_{j,k} \ge C_{j,k-1}$$
3. **Machine Exclusivity (Capacity One):** For every pair of distinct operations $u \neq v \in O$ sharing a machine ($\mu(u) = \mu(v)$), processing intervals cannot overlap:
   $$C_u \le S_v \quad \lor \quad C_v \le S_u$$

### 2.3 Objective Function
The primary objective is minimizing **makespan** ($C_{\max}$), defined as the maximum completion time across all jobs (equivalently across all operations):
$$\min C_{\max} = \min \max_{j \in J} C_j = \min \max_{o \in O} C_o$$
where $C_j = C_{j, m_j - 1}$ is the completion time of job $j$'s final operation.

---

## 3. Shared Representation, Decoder & Schedule-Neighborhood Adapter Contract

All primary comparisons across Phases A, B, and C share one immutable representation and evaluation contract implemented solely in Rust (`crates/jssp-core`):

### 3.1 Operation-Based Representation (OBR)
- Chromosome length is fixed at $L = \sum_{j=0}^{n-1} m_j$.
- A valid OBR chromosome $\pi = (\pi_0, \pi_1, \dots, \pi_{L-1}) \in J^L$ contains each job identifier $j \in \{0, \dots, n-1\}$ **exactly $m_j$ times**.
- The $k$-th occurrence (0-indexed) of job $j$ scanning left-to-right uniquely identifies operation $O_{j,k}$.
- **Validation invariant:** Length $= L$, all IDs within $0 \le \pi_t < n$, and exact per-job multiplicity $\text{count}(j, \pi) = m_j$.

### 3.2 Canonical Deterministic Serial Insertion Decoder
1. Scan $\pi$ strictly from $t = 0$ to $L - 1$.
2. For $\pi_t = j$, look up its next unscheduled operation $o = O_{j,k}$ and machine $r = \mu(o)$.
3. Determine earliest feasible start time $S_o \ge C_{j,k-1}$ (with $C_{j,-1} = 0$) respecting machine availability on $r$ under the declared insertion/earliest-feasible-gap policy (active/semi-active gap insertion rules to be finalized in `docs/algorithms/obr_decoder.md` and `configs/defaults/decoder.toml`).
4. Resolve all equal-time ties deterministically (e.g., by operation/job ID or chromosome index; never via randomized or hash-map order).
5. Return operation start/completion times $(S_o, C_o)$, per-machine operation sequences, disjunctive graph representation, and makespan $C_{\max}$.

### 3.3 Schedule-Neighborhood Adapter (Mandatory for N5–N8)
Because N5–N8 operate on critical blocks and machine-disjunctive arcs rather than arbitrary OBR gene swaps:
1. Decode OBR $\pi$ to the canonical schedule and disjunctive graph $D(\pi)$.
2. Identify critical paths / critical blocks and apply the exact published N5–N8 move to machine sequences.
3. Validate job precedence, machine exclusivity, and disjunctive graph acyclicity (rejecting cyclic graphs).
4. Evaluate exact candidate makespan $C_{\max}$ (with an exact reference evaluator verifying any incremental evaluation).
5. Re-encode a retained neighbor graph back to an OBR chromosome $\pi'$ via a stable topological sort with documented tie-breaking.
6. Re-decode $\pi'$ and assert that the intended machine sequences/constraints are preserved; reject and log any move whose machine order cannot be faithfully preserved in OBR.

---

## 4. Experimental Phases & Configuration Matrix

The research design comprises **67 planned configurations** across three sequential/modular phases (**24 + 32 + 11 = 67**), or **63 configurations** if standalone Metropolis–Hastings (`B25`–`B28`) is shown to be mathematically redundant with Simulated Annealing (`B21`–`B24`).

**No Cartesian Explosion:** Phases do **not** run the full $24 \times 32 \times 11$ product. Development/screening instances select representative fixed components between phases prior to frozen confirmatory evaluation.

### 4.1 Phase A — Initial Population Generation (24 Configurations: `A01`–`A24`)

Phase A isolates the schedule/population initializer while keeping OBR, population size, canonical decoder, and downstream GA parameters constant.

For MWR-biased sequential construction, at construction step $t$, let $R$ be the set of jobs with at least one remaining unscheduled operation, and $W_j$ be the total remaining processing time of job $j$:
$$P(j) = \frac{W_j^{\alpha}}{\sum_{k \in R} W_k^{\alpha}}$$
where $\alpha = 0$ corresponds to uniform selection (control `A01`/`A04`–`A07`), $\alpha = 1.0$ is the primary MWR exponent, and $\alpha \in \{0.5, 2.0\}$ provides sensitivity bounds.

| ID | Initializer Family | MWR Bias ($\alpha$) / Population Mix | Category |
|---|---|---:|---|
| `A01` | Uniform Random | — ($\alpha = 0$) | Baseline |
| `A02` | Deterministic MWR | Deterministic rule | Deterministic control |
| `A03` | Monte Carlo + MWR bias | $\alpha = 1.0$ | Primary MC+MWR |
| `A04` | Halton | — | Unscrambled QMC |
| `A05` | Scrambled Halton | — | Randomized QMC |
| `A06` | Sobol | — | Unscrambled QMC |
| `A07` | Scrambled Sobol | — | Randomized QMC |
| `A08` | Halton + MWR bias | $\alpha = 1.0$ | Primary QMC+MWR |
| `A09` | Scrambled Halton + MWR bias | $\alpha = 1.0$ | Primary RQMC+MWR |
| `A10` | Sobol + MWR bias | $\alpha = 1.0$ | Primary QMC+MWR |
| `A11` | Scrambled Sobol + MWR bias | $\alpha = 1.0$ | Primary RQMC+MWR |
| `A12` | Mixed Halton / Sobol | 50% Halton : 50% Sobol | Mixed QMC population |
| `A13` | Mixed Halton / Sobol | 25% Halton : 75% Sobol | Mixed QMC population |
| `A14` | Mixed Halton / Sobol | 75% Halton : 25% Sobol | Mixed QMC population |
| `A15` | Monte Carlo + MWR bias | $\alpha = 0.5$ | MWR sensitivity |
| `A16` | Monte Carlo + MWR bias | $\alpha = 2.0$ | MWR sensitivity |
| `A17` | Halton + MWR bias | $\alpha = 0.5$ | MWR sensitivity |
| `A18` | Halton + MWR bias | $\alpha = 2.0$ | MWR sensitivity |
| `A19` | Scrambled Halton + MWR bias | $\alpha = 0.5$ | MWR sensitivity |
| `A20` | Scrambled Halton + MWR bias | $\alpha = 2.0$ | MWR sensitivity |
| `A21` | Sobol + MWR bias | $\alpha = 0.5$ | MWR sensitivity |
| `A22` | Sobol + MWR bias | $\alpha = 2.0$ | MWR sensitivity |
| `A23` | Scrambled Sobol + MWR bias | $\alpha = 0.5$ | MWR sensitivity |
| `A24` | Scrambled Sobol + MWR bias | $\alpha = 2.0$ | MWR sensitivity |

**Phase A Scientific & Algorithmic Invariants:**
1. **Discrete QMC Mapping:** Each continuous QMC coordinate $u \in [0, 1)$ must map to an eligible job in $R$ via a documented sequential mapping (e.g., deterministic canonical ordering of $R$ followed by inverse-CDF partitioning using $P(j)$ or $1/|R|$). Do **not** assume continuous low-discrepancy guarantees survive this adaptive discrete mapping; measure actual population diversity, duplicate rate, and initial $C_{\max}$.
2. **Coordinate Consumption & Sobol Constraints:** Document exact dimensionality ($d = L$, or dimension truncation/folding strategy for large $L$), Sobol direction numbers / maximum supported dimension, and power-of-two sample-size constraints.
3. **Scrambling & Mixing:** Use documented scrambling methods (e.g., Owen-style / random digit shift for Sobol, Faure/random-digit scrambling for Halton) with independent scramble RNG streams. Mixed Halton/Sobol (`A12`–`A14`) assigns subpopulations to each generator at exact proportions—never by averaging or interleaving raw coordinates.
4. **Deterministic MWR (`A02`):** Must use a fixed deterministic tie-breaker (e.g., lowest job ID); any duplicate schedules produced across a population must be reported honestly as zero-diversity behavior.

---

### 4.2 Phase B — Local Search & Metaheuristics (32 Planned Configurations: `B01`–`B32`; 28 if `B25`–`B28` Removed)

Phase B crosses 4 schedule neighborhood structures ($\mathcal{N} = \{\text{N5}, \text{N6}, \text{N7}, \text{N8}\}$) with 8 search strategies:

| Search Strategy | N5 | N6 | N7 | N8 | Primary Strategy Notes |
|---|:---:|:---:|:---:|:---:|---|
| **Steepest Descent** | `B01` | `B02` | `B03` | `B04` | Evaluates full feasible neighborhood; selects strict best improvement. Cannot claim local optimality if budget interrupts scan. |
| **First Improvement** | `B05` | `B06` | `B07` | `B08` | Deterministic documented scan order; accepts first strict improvement. |
| **Random Improvement** | `B09` | `B10` | `B11` | `B12` | Samples/selects improving moves using a dedicated RNG stream. |
| **Tabu Search** | `B13` | `B14` | `B15` | `B16` | Tabu list tenure, move attribute, and aspiration criteria must be specified and frozen pre-confirmatory. |
| **Iterated Local Search (ILS)** | `B17` | `B18` | `B19` | `B20` | Fixed perturbation kernel, strength, acceptance criterion, and embedded Steepest Descent local search. |
| **Simulated Annealing (SA)** | `B21` | `B22` | `B23` | `B24` | Cooling temperature schedule $T_t$, Metropolis acceptance, and development-tuned schedule parameters. |
| **Standalone Metropolis–Hastings (MH)** *(Contingent)* | `B25` | `B26` | `B27` | `B28` | Fixed target $\pi(S) \propto \exp(-\beta C_{\max}(S))$ with explicit forward/reverse proposal $q(S' \mid S)$ and Hastings correction $\alpha(S,S')$. **Remove if redundant with SA.** |
| **Hybrid ILS + MH** | `B29` | `B30` | `B31` | `B32` | MH filter ($\alpha_{\text{MH}}$ on perturbation kernel) gates raw ILS perturbation proposal *before* deterministic local search descent. |

**Phase B Scientific & Algorithmic Invariants:**
1. **Source-Verified N5–N8:** N5–N8 have non-uniform definitions across literature (e.g., Nowicki–Smutnicki N5, Balas–Vazacopoulos N6, Zhang et al. N7, Kuhpfahl–Bierwirth / Xie et al. N8). Exact move generation rules, block-boundary preconditions, and acyclicity guarantees must be documented with verified citations in `docs/algorithms/n5_n6_n7_n8_definitions.md` before implementation.
2. **SA vs. Standalone MH (`B25`–`B28`) Distinction Criterion:**
   - Standalone MH must define a **time-homogeneous fixed target distribution** $\pi(S)$ (e.g., $\pi(S) \propto \exp(-\beta C_{\max}(S))$ with fixed inverse temperature $\beta > 0$) and compute exact forward/reverse proposal probabilities $q(S' \mid S)$ and $q(S \mid S')$ for the chosen neighborhood (vital because critical-block neighborhoods are generally **asymmetric**: $|\mathcal{N}(S)| \neq |\mathcal{N}(S')|$):
     $$\alpha(S, S') = \min\left(1, \frac{\pi(S')\, q(S \mid S')}{\pi(S)\, q(S' \mid S)}\right)$$
   - If the implemented MH reduces to symmetric acceptance without Hastings correction or duplicates SA's cooling schedule and proposal kernel, `B25`–`B28` **must be deleted** and the total configuration count revised from **67 to 63**.
3. **Hybrid ILS + MH (`B29`–`B32`) Semantics:** MH acceptance operates strictly on the perturbation transition $S \to S_{\text{pert}}$ prior to deterministic inner local search $\text{LS}(S_{\text{pert}})$. Detailed balance is claimed **only** if mathematically established for the perturbation kernel alone—never for the full perturbation-plus-descent transition.

---

### 4.3 Phase C — Memetic Genetic Algorithms (11 Configurations: `C01`–`C11`)

Phase C evaluates classical GA, standard memetic learning rules, and the proposed persistent paired **Dual-View** architecture under identical OBR genetic operators (selection, crossover, mutation, elitism) and equalized total evaluation/genotype budgets:

| ID | Method | Genotype / Phenotype / Gate Specification | Role |
|---|---|---|---|
| `C01` | Classical GA | Non-memetic OBR GA (no local search) | Baseline |
| `C02` | Lamarckian GA (LGA) | Local-search-improved phenotype is written back into genotype | Memetic baseline |
| `C03` | Baldwinian GA (BGA) | Original genotype retained; fitness uses locally improved phenotype | Memetic baseline |
| `C04` | Partial Lamarckian GA (PLGA) | 50% Lamarckian write-back rate (selection rule frozen pre-confirmatory) | Memetic baseline |
| `C05` | Dual-View Fixed Gate ($p = 0.5$) | Paired $(ID_i, G_i^L, G_i^B)$; constant equal gate weight $p = 0.5$ | Primary Dual-View |
| `C06` | Dual-View Adaptive Gate | Paired views with deterministic rule-based gate $p_t$ using performance & diversity | Primary Dual-View |
| `C07` | Dual-View Fixed Gate ($p = 0.25$) | Sensitivity check favoring Baldwinian view | Sensitivity |
| `C08` | Dual-View Fixed Gate ($p = 0.75$) | Sensitivity check favoring Lamarckian view | Sensitivity |
| `C09` | Adaptive Gate without diversity signal | Ablation removing diversity feedback from adaptive gate update | Ablation |
| `C10` | Unpaired Dual-Population Control | Two independent sub-populations (one LGA, one BGA) without shared $ID_i$ pairing | Control |
| `C11` | Controlled Cross-View Migration | Exploratory ablation adding fixed-rate transfer between $G_i^L$ and $G_i^B$ | Exploratory ablation |

**Proposed Dual-View Architecture Specification:**
- **Paired Representation:** Each individual maintains a shared identity and two distinct genotypes initialized from the same starting OBR chromosome:
  $$I_i = (\mathrm{ID}_i, G_i^L, G_i^B)$$
  - **Lamarckian View ($G_i^L$):** Local search modifies phenotype and writes the improved OBR back into $G_i^L$.
  - **Baldwinian View ($G_i^B$):** Local search evaluates the improved phenotype for fitness, while $G_i^B$ remains untouched.
- **Lineage Isolation:** In `C05`–`C09`, crossover and mutation occur strictly **within-view** ($G^L \times G^L$ and $G^B \times G^B$). Cross-view transfer is isolated solely in exploratory configuration `C11`.
- **Gate & Pair Selection:** Both views remain active in every generation; the gate weight $p_t \in [p_{\min}, p_{\max}] \subseteq [0, 1]$ governs selection/reproductive allocation. A candidate pair-level score is:
  $$S_i(t) = p_t S_i^L + (1 - p_t) S_i^B$$
  Exact score normalization, tie-breaking, reproductive slot allocation, update interval, and the rule-based adaptive update equation using performance and diversity signals must be fully specified in `docs/algorithms/dual_view.md` before implementation.
- **Budget & Capacity Parity:** Comparisons between single-view GAs (`C01`–`C04`) and dual-view/dual-population variants (`C05`–`C11`) must equalize **both total genotype capacity** (e.g., $N$ single-view genotypes vs. $N/2$ Dual-View pairs $= N$ total genotypes) and **total objective evaluations** (counting both initial evaluations and all embedded local-search evaluations across both views).

### 4.4 Checked-In Configuration Matrix Status & Discrepancies
- Expected total configurations: **24 (Phase A) + 32 (Phase B) + 11 (Phase C) = 67** (or **63** if `B25`–`B28` are removed).
- **Actual checked-in config status:** All 29 TOML files in `configs/` (`defaults/`, `phase_a/`, `phase_b/`, `phase_c/`, `matrix/`, `benchmarks/`, `protocols/`) are **0-byte empty placeholders**. Zero configuration IDs are presently encoded in TOML. Populating and validating these TOMLs against a Pydantic schema is a prerequisite for M2/M3.

---

## 5. Benchmark Hierarchy & Provenance Policy

Benchmark tracks are strictly separated by instance source, scale, and scheduling model:

| Track | Dataset / Family | Instance Count | Model Class | Usage & Reporting Rule |
|---|---|---:|---|---|
| **Track 1** | **Core Historical Rectangular JSSP** (FT 3, LA 40, ABZ 5, ORB 10, SWV 20, YN 4, TA 80, DMU 80) | **242** | Standard rectangular $n \times m$ | Primary classical baseline; report stratified by family and proven/BKS status |
| **Track 2a** | **Large-TA Rectangular Extension** (Da Col & Teppan; 8 strata below $1000\times 1000$, 10 each) | **80** | Standard rectangular $n \times m$ | Extended scale generalization for shortlisted methods |
| **Track 2b** | **Extreme Stress Track** (Da Col & Teppan $1000 \times 1000$ TAI instances) | **10** | Standard rectangular ($10^6$ ops) | Separate extreme-scale shortlist under dedicated time/memory caps ($80 + 10 = 90$ Large-TA total) |
| **Track 3** | **Strassl–Musliu Synthetic Generalization** | **100–200** (stratified from 4,225) | Standard rectangular | Stratified by size, $n/m$ ratio, and duration distribution; split into dev vs. held-out test |
| **Track 4** | **Da Col–Teppan Variable-Route / Known-Optima** | **24** | Variable route / potential re-entrant | **Excluded from primary OBR track** unless adapter explicitly handles variable routing; report separately |
| **Track 5** | **BEL Industrial Extension** *(Optional)* | **20** | Industrial scheduling | Include **only** after verifying file format and recomputing $C_{\max}$; never compare against original weighted-completion-time scores |

**Provenance & Bounds Policy:**
- No benchmark instance files, checksums, or BKS tables exist yet in `data/` (only 5 `.gitignore` files exist).
- Every downloaded benchmark file must store source URL, license/usage terms, retrieval timestamp, SHA-256 checksum, and format notes in `data/metadata/`.
- Best-known upper bounds (BKS) and lower bounds (LB) sourced from JSPLib or primary literature must be snapshotted with date-stamps and explicit optimality-proven flags (`optimal` vs. `open_ub`) in `data/bounds/`. Frozen snapshots must never be silently mutated during or after confirmatory experiments.

---

## 6. Fairness Protocol & Computational Budgets

1. **Single Authoritative Semantics:** All methods use the exact same Rust instance loader, OBR validator, deterministic serial decoder, feasibility checker, and makespan evaluator.
2. **Equal Objective-Evaluation Budgets:**
   - **Phase A:** Identical population size $N$ across `A01`–`A24`, and identical downstream GA evaluation budget when evaluating post-initialization convergence. Initial population metrics (makespan distribution, diversity, duplicate rate, construction runtime) are logged separately from GA search.
   - **Phase B:** Matched starting schedules (paired instance-seed blocks) and identical maximum objective-evaluation budget per run. Record generated moves, feasible moves, rejected adapter moves, improving moves, accepted moves, and wall-clock runtime.
   - **Phase C:** Identical baseline GA parameters (selection, crossover, mutation, elitism) and identical total genotype count + total schedule/local-search evaluation budget. Dual-View local search costs on both $G^L$ and $G^B$ are fully charged to the shared run budget.
3. **Randomness & Reproducibility:**
   - Each run records a 64-bit/128-bit `root_seed` and derives isolated, named independent RNG substreams (`init`, `qmc_scramble`, `selection`, `crossover`, `mutation`, `neighborhood_sample`, `perturbation`, `sa_accept`, `mh_accept`). Changing a parameter affecting one stream must not shift downstream random draws in other streams.
   - Confirmatory protocol target: **20 independent seeds** per stochastic configuration/instance (subject to pilot runtime verification and frozen protocol lock). Deterministic configurations (`A02`, unscrambled `A04`/`A06`/`A08`/`A10`, `B01`–`B08`) run once per paired start unless randomized tie-breaking or randomized starts apply.
4. **4-Stage Experimental Separation:**
   - *Stage 1 (Correctness Pilot):* Tiny/medium hand-verified fixtures (`tests/fixtures/`).
   - *Stage 2 (Development / Screening):* Stratified development split only—used to tune SA cooling schedules, Tabu tenures, ILS perturbation strengths, MH $\beta$, and Adaptive Gate parameters.
   - *Stage 3 (Frozen Confirmatory):* Freeze Git commit SHA, `uv.lock`, `Cargo.lock`, TOML configs, seed list, and held-out benchmark split before executing confirmatory runs. Never tune on held-out test instances.
   - *Stage 4 (Scalability / Extension):* Run shortlisted methods on Tracks 2a, 2b, 4, and 5 and report separately.
5. **Immutable Run Manifests:** Every run writes an immutable JSON manifest capturing configuration/phase ID, resolved parameters, instance ID & SHA-256, root & substream seeds, Git SHA + dirty status, Rust/Python/dependency versions, hardware/CPU/OS/build-profile metadata, budget limits, timestamps, feasibility status, and algorithm-specific telemetry.

---

## 7. Statistical Evaluation Plan

### 7.1 Recorded Metrics
- **Solution Quality:** Best, mean, median, IQR, and standard deviation of $C_{\max}$; **Relative Percentage Deviation (RPD)** relative to the frozen BKS snapshot:
  $$\mathrm{RPD}(\%) = 100 \times \frac{C_{\max} - \mathrm{BKS}}{\mathrm{BKS}}$$
- **Reliability & Effort:** Feasible-schedule rate, target/BKS hit rate, total objective evaluations, move-generation/feasibility/improvement/acceptance counts, wall-clock seconds, and anytime $C_{\max}$-vs-evaluations trajectories.
- **Population & Memetic Telemetry:** Genotypic diversity (e.g., pairwise normalized Hamming/position distance), unique chromosome ratio / duplicate rate, Dual-View gate trajectory $p_t$, per-view best $C_{\max}$, per-view local-search evaluation split, and migration count (`C11`).

### 7.2 Statistical Inference Protocol
- **Unit of Analysis & Stratification:** Report instance-level and benchmark-family-stratified results (avoid hiding poor performance on hard families like SWV/DMU/TA behind a grand average dominated by easy FT/LA instances).
- **Non-Parametric Blocked Tests:** Because makespan and RPD distributions across instances are non-normal and paired by instance/seed:
  - *Multi-method omnibus test:* Friedman test across aligned instance-level summary blocks (e.g., median RPD across 20 seeds per instance).
  - *Pairwise post-hoc comparisons:* Two-sided Wilcoxon signed-rank tests (paired by instance or paired instance-seed blocks) with **Holm–Bonferroni** family-wise error rate correction.
  - *Effect Sizes & Intervals:* Report non-parametric effect sizes (Vargha–Delaney $\hat{A}_{12}$ or Cliff's $\delta$) and bootstrap confidence intervals alongside adjusted $p$-values.

---

## 8. Related-Work Positioning

To maintain strict research integrity, the project explicitly distinguishes prior published methods from the proposed Dual-View hypothesis:
- **Neighborhoods N5–N8:** Established critical-path neighborhoods from scheduling literature; N8 is published work and is evaluated rather than claimed as an invention.
- **QMC Initialization:** Halton, Sobol, and randomized QMC sequences have prior applications in scheduling and metaheuristics; this study empirically evaluates their behavior under discrete sequential MWR-weighted OBR mapping without assuming continuous discrepancy preservation.
- **Adaptive Memetic Learning:** Prior literature explores probabilistic/adaptive switching between Lamarckian and Baldwinian operators at the population or individual-event level. The proposed **Dual-View mechanism** specifically hypothesizes that maintaining **two persistent, identity-paired genotype lineages** ($I_i = (\mathrm{ID}_i, G_i^L, G_i^B)$) evolving within-view under a shared pair-level gate outperforms single-view memetic GAs (`C02`–`C04`) and unpaired dual populations (`C10`). This remains an open empirical hypothesis until verified by controlled ablations and systematic literature review.

---

## 9. Known Limitations & Pending Scientific Decisions

### 9.1 Known Limitations
- **Adaptive Discrete QMC Mapping:** Mapping continuous QMC vectors through state-dependent eligible-job sets $R$ with MWR probabilities breaks standard Koksma–Hlawka uniformity guarantees; empirical diversity analysis is required.
- **OBR Adapter Coverage:** Certain published moves on disjunctive graphs or machine sequences may induce precedence/OBR re-encoding ambiguities; moves that cannot preserve the exact machine-order semantics on round-trip must be rejected and counted honestly.
- **Hardware Sensitivity of Wall-Clock Time:** Objective evaluations are deterministic across platforms for identical builds, whereas wall-clock runtimes depend on CPU architecture, compiler flags, and system load and must not rely on CI runner timings.
- **Cross-Track Incomparability:** Variable-route (`Track 4`) and industrial (`Track 5`) instances cannot share the rectangular OBR baseline without explicit model extensions.

### 9.2 Pending Scientific Decisions (`D01`–`D07`) Requiring Resolution Before Confirmatory Runs
| ID | Decision Topic | What Must Be Decided & Frozen | Target Document |
|---|---|---|---|
| `D01` | **Canonical Decoder Gap Policy** | Active (earliest feasible idle gap insertion) vs. non-delay/semi-active insertion and exact equal-time tie-break rule | `docs/algorithms/obr_decoder.md` |
| `D02` | **QMC Discrete Mapping & Sobol Bounds** | Exact sequential inverse-CDF mapping, eligible-job ordering rule, coordinate consumption for dimension $L > d_{\max}$, scrambling algorithm, and Sobol $2^k$ population rounding rule | `docs/algorithms/qmc_mapping.md` |
| `D03` | **Authoritative N5–N8 Citations & Rules** | Exact published source papers, block-boundary rules, feasibility conditions, and OBR round-trip topological tie-breaker for N5, N6, N7, and N8 | `docs/algorithms/n5_n6_n7_n8_definitions.md` |
| `D04` | **SA vs. Standalone MH Distinction** | Fixed target $\beta$, exact asymmetric Hastings proposal ratio $q(S \mid S') / q(S' \mid S)$, and formal decision whether to retain `B25`–`B28` (67 configs) or prune as redundant with SA (63 configs) | `docs/algorithms/mh_vs_sa.md` |
| `D05` | **Phase B Hyperparameters & Budgets** | Default inner local search for ILS (`Steepest` vs `First`), perturbation operator/strength, Tabu tenure/aspiration, SA cooling schedule, and evaluation budgets | `configs/phase_b/*.toml`, `docs/experiments/phase_b.md` |
| `D06` | **Dual-View Gate & Reproduction Math** | Exact score normalization ($S_i^L, S_i^B$), reproductive slot allocation rule, adaptive gate update equation ($p_t$), update window, diversity metric, $[p_{\min}, p_{\max}]$ bounds, PLGA 50% selection rule, and `C11` migration protocol | `docs/algorithms/dual_view.md`, `configs/phase_c/dual_view.toml` |
| `D07` | **Baseline GA, Seed Count & Benchmark Splits** | Frozen classical GA parameters (population size $N$, crossover/mutation operators & rates, tournament size, elitism), development vs. held-out split definitions, and Track 4/5 inclusion status | `docs/experiments/fairness_protocol.md`, `configs/protocols/*.toml` |
