# Fairness, Reproducibility, Benchmark Splits & Statistical Protocol (`D07`)

**Decision ID:** `D07` (`ACCEPTED`)
**Scope:** Single-source evaluation guarantees, named RNG stream hierarchy, seed schedules, benchmark split isolation (Tracks 1–5), exact metric formulas, and statistical testing protocol.

---

## 1. Common Evaluation & Fairness Invariants

1. **Authoritative Rust Semantics:** All configurations (`A01`–`A24`, `B01`–`B24`, `B29`–`B32`, `C01`–`C11`) use the identical Rust instance representation, OBR validator, Serial SGS active decoder (`u64` integer times), feasibility checker, and exact $C_{\max}$ evaluator (`crates/jssp-core`).
2. **Equal Objective-Evaluation Budgets:** Every Serial SGS decode and every disjunctive-graph neighborhood evaluation increments one unified `eval_counter`. Run termination enforces `eval_counter <= B_eval` cleanly across single-view and Dual-View algorithms (`C05`–`C11` count both $G^L$ and $G^B$ and all local search calls).
3. **Iteration vs. Effort:** Iteration/generation counts are logged diagnostically only and are **never** used as an equal-effort stopping rule. Wall-clock time and hardware metadata are recorded separately.

---

## 2. Seed Schedule & Isolated Named PRNG Substreams

- **Master Seed Schedule (20 Confirmatory Seeds):**
  Each stochastic configuration executes **$R = 20$** independent runs per instance using fixed deterministic master seeds derived via SHA-256 hash truncation of `"jssp-research-v1-seed-{r}"` for $r \in \{0, \dots, 19\}$ (or explicit deterministic array `[101, 202, ..., 2020]` frozen in `configs/protocols/confirmatory.toml`). *Justification for $R = 20$:* Provides $19$ degrees of freedom for instance-level median/IQR estimation and sufficient paired block resolution across $194$ confirmatory Track 1 instances while keeping $63 \times 194 \times 20$ runs tractable in native Rust.
- **Named Substreams (`crates/jssp-core/src/rng/`):** From `master_seed`, derive 9 independent `ChaCha8Rng` / `SplitMix64`-seeded substreams tagged by exact ASCII domain strings:
  1. `"init"` (random OBR / MC-MWR)
  2. `"qmc_scramble"` (Halton digit permutations / Sobol Matoušek scrambling)
  3. `"selection"` (tournament draws)
  4. `"crossover"` (POX job subset partition)
  5. `"mutation"` (swap/insert locus & operator choice)
  6. `"local_search_sample"` (Random Improvement / SA neighbor sampling / Tabu tenure)
  7. `"perturbation"` (ILS / TS diversification swaps)
  8. `"sa_accept"` (Simulated Annealing Metropolis coin flips)
  9. `"mh_filter"` (ILS+MH pre-descent perturbation filter coin flips)
- **Stream Independence Invariant:** Modifying a parameter or draw count in one substream (e.g., mutation rate) **must never alter** random draws consumed by any other substream (e.g., crossover or initialization). Verified by dedicated M2 unit tests.

---

## 3. Benchmark Hierarchy, Splits & Track Isolation (`D07-B`, `D07-C`)

| Track | Dataset & Citation | Total | Development / Screening | Held-Out Confirmatory / Test | Track Policy |
|---|---|---:|---:|---:|---|
| **Track 1** | **Core Historical Rectangular** (FT 3, LA 40, ABZ 5, ORB 10, SWV 20, YN 4, TA 80 [Taillard 1993], DMU 80 [Demirkol et al. 1998]) | **242** | **48** ($\approx 20\%$ family-stratified) | **194** ($\approx 80\%$ family-stratified) | Primary classical benchmark. Dev split includes representative sizes per family (`FT06` in pilot; `TA`/`DMU`/`LA`/`SWV` strata balanced). |
| **Track 2a** | **Large-TA Rectangular Extension** (Da Col & Teppan 2019, 8 strata $< 1000\times 1000$) | **80** | — | **80** (shortlisted methods) | Scalability generalization; note standard Sobol (`d_max = 21,201`) applies only to strata with $L \le 21,201$. |
| **Track 2b** | **Extreme Stress Track** (Da Col & Teppan 2019, $1000\times 1000$) | **10** | — | **10** (shortlist + wall-clock/RAM cap) | Isolated stress benchmark; standard Sobol excluded (`L = 10^6`). |
| **Track 3** | **Strassl–Musliu Synthetic Generalization** (2022, from 4,225 instances) | **160** | **40** (stratified by $n, m, n/m$, duration dist.) | **120** (stratified held-out) | Prevents overfitting to classic synthetic distributions; preserves original metadata/splits. |
| **Track 4** | **Da Col–Teppan Variable-Route / Known-Optima** (2019) | **24** | — | — | **DEFERRED (`D07-C`):** variable route lengths / potential machine revisits violate rectangular OBR assumption. |
| **Track 5** | **BEL Industrial Extension** | **20** | — | — | **DEFERRED (`D07-C`):** requires machine-format verification and $C_{\max}$ recomputation (never compare against $\sum w_j C_j$). |

**Anti-Leakage Rule:** All parameter tuning (SA cooling check, adaptive gate bounds/weights check, screening shortlist) uses **only** Track 1 Dev (`48`) and Track 3 Dev (`40`). Held-out instances (`194` + `120`) remain locked until `configs/`, `Cargo.lock`, `uv.lock`, and Git commit SHA are frozen.

---

## 4. Exact Evaluation Metric Formulas

For instance $k$ with frozen JSPLib/literature reference Best-Known Upper Bound $\mathrm{BKS}_k$ and Lower Bound $\mathrm{LB}_k$ (from date-stamped `data/bounds/` snapshot):
- **Makespan Statistics (over $R$ seeds):** Best $C_{\max}^{\min}$, Mean $\bar{C}_{\max}$, Median $\tilde{C}_{\max}$, Standard Deviation $s_{C_{\max}}$, Interquartile Range $\mathrm{IQR}$.
- **Relative Percentage Deviation ($\mathrm{RPD}_{\mathrm{BKS}}$):**
  $$\mathrm{RPD}_{\mathrm{BKS}}(\%) = 100 \times \frac{C_{\max} - \mathrm{BKS}_k}{\mathrm{BKS}_k}$$
- **Lower-Bound Optimality Gap ($\mathrm{Gap}_{\mathrm{LB}}$):**
  $$\mathrm{Gap}_{\mathrm{LB}}(\%) = 100 \times \frac{C_{\max} - \mathrm{LB}_k}{\mathrm{LB}_k}$$
- **Success / Hit Rate:** Fraction of runs achieving $C_{\max} \le \mathrm{BKS}_k$ (or proven optimum when $\mathrm{BKS}_k = \mathrm{LB}_k$).
- **Normalized Pairwise OBR Hamming Diversity:**
  $$D_{\text{Ham}}(P) = \frac{2}{N(N - 1)} \sum_{0 \le i < j < N} \frac{1}{L} \sum_{\ell=0}^{L-1} \mathbb{I}\!\big[\pi_i[\ell] \neq \pi_j[\ell]\big]$$
- **Duplicate Rate:** $\mathrm{Dup}_{\text{OBR}} = 1 - |\text{unique}(\pi_0, \dots, \pi_{N-1})| / N$ (and analogously $\mathrm{Dup}_{\text{sched}}$ over unique machine-sequence canonical hashes).
- **Anytime Performance (Normalized Area Under $C_{\max}$ Curve):** Trapezoidal area of best-so-far $\mathrm{RPD}_{\mathrm{BKS}}(e)$ over normalized evaluation fraction $e / B_{\text{eval}} \in [0, 1]$.

---

## 5. Statistical Inference Plan

- **Stratification:** Always report results stratified by benchmark family (`FT`, `LA`, `ABZ`, `ORB`, `SWV`, `YN`, `TA`, `DMU`, Track 2a, Track 2b, Track 3) alongside overall paired ranks.
- **Omnibus Multi-Method Test:** **Friedman test** blocked by instance (using per-instance median $\mathrm{RPD}_{\mathrm{BKS}}$ across the $20$ seeds) at $\alpha = 0.05$.
- **Post-Hoc Pairwise Comparisons:** Two-sided **Wilcoxon signed-rank test** paired by `(instance)` (or `(instance, seed)` for matched-start Phase B/C pairs).
- **Multiple-Comparison Correction:** **Holm–Bonferroni** step-down adjustment applied within each phase's planned contrast family.
- **Effect Sizes & Uncertainty:** Report **Vargha–Delaney $\hat{A}_{12}$** (stochastic superiority probability) and/or **Cliff's $\delta$**, plus $95\%$ bias-corrected bootstrap confidence intervals for median RPD differences.

---

## 6. Immutable Run Manifest Schema Requirements
Every run emits a JSON manifest containing: `experiment_id`, `phase`, `protocol_stage`, `resolved_params`, `benchmark_track`, `instance_id`, `instance_sha256`, `bks_snapshot_id`, `bks_value`, `lb_value`, `is_proven_optimal`, `run_id`, `master_seed`, `substream_seeds`, `git_commit`, `git_dirty`, `toolchain_versions` (Python, uv, Rust, Cargo, PyO3, Maturin, NumPy, pandas, SciPy, Matplotlib), `hardware` (OS, CPU, cores, RAM, build profile), `eval_budget`, `wall_limit_s`, `start_utc`, `end_utc`, `elapsed_s`, `evals_used`, `cmax`, `rpd_bks`, `feasible`, `terminated_reason`, and phase-specific telemetry.
