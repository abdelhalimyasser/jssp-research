# Exploratory Protocol — Dual-View Local Search Budget Investigation

- **Status:** Exploratory Extension (`exploratory/dual-view-ls-budget`, **strictly separate from frozen confirmatory Phase C `C01`–`C11` and final pipeline `P1`–`P5`**)
- **Objective:** Evaluate whether adjusting local-search evaluation allocation inside the Model A Dual-View architecture (`C06`, `A11` initialization, `N5` neighborhood, equalized total fitness-evaluation budget $B_{\text{GA}} = \mathrm{clamp}(64\lceil 2.5JM\rceil, 6400, 64000)$) alters search dynamics and solution quality without altering GA operators, population capacity (`N = 64`, `32` pairs), or budget parity.

---

## 1. Pre-Modification Code & Telemetry Audit

Inspecting [`EvalCounter`](file:///home/abdelhalimyasser/Downloads/jssp-research/crates/jssp-core/src/telemetry/counters.rs), [`decode_and_local_search`](file:///home/abdelhalimyasser/Downloads/jssp-research/crates/jssp-core/src/memetic/mod.rs), [`run_dual_view_model_a`](file:///home/abdelhalimyasser/Downloads/jssp-research/crates/jssp-core/src/memetic/dual_view/mod.rs), and [`run_simulated_annealing`](file:///home/abdelhalimyasser/Downloads/jssp-research/crates/jssp-core/src/metaheuristics/simulated_annealing.rs) shows:

1. **Evaluation Conservation:** `EvalCounter` strictly enforces `evals_used == init_evals + search_evals == decode_evals + ls_evals <= evaluation_budget`. Neither branch bypasses the shared counter.
2. **Root Mechanisms of `96.9%`–`97.5%` LS Share and ~`12` Generations:**
   - Every decoded pair (`32` pairs at $t=0$; `31` non-elite offspring pairs at $t \ge 1$) applies bounded Steepest Descent (`N5`, $b_{\text{LS}} = 50$) to **both** views (`62`–`64` LS calls/generation).
   - At ~`33.9` LS evaluations per decode (`32.18` Dev / `35.70` Held-Out in `C06`), one generation consumes `~2,164` evaluations under a mean $B_{\text{GA}}$ of `29,954`, limiting runs to `13.15` Dev / `11.95` Held-Out generations.
   - At $t = 0$, $G^L(0) = G^B(0) = \pi_i^{(0)}$, so deterministic Steepest Descent executes twice on identical initial schedules.
   - Because Baldwinian genotypes are never written back, post-crossover Baldwinian offspring consume more LS evaluations per decode while producing better branch best makespan in `< 9%` of Held-Out `C06` runs.
   - Standalone `run_simulated_annealing` performs `50` uncommitted calibration probes before cooling (`epoch_len = 2L`), making it incompatible with $b_{\text{LS}} \le 50$ memetic micro-calls unless adapted into a short-horizon budget-limited kernel.

---

## 2. Controlled Sequential Screening Design (`16` Dev $\times$ `20` Seeds $\to$ `16` Held-Out)

All experiments preserve `A11` initialization, `N5` critical-block neighborhood, `C06` Model A adaptive gate ($w_{\text{perf}} = 0.70, w_{\text{div}} = 0.30, p_0 = 0.50$), `N = 64` genotypes (`32` pairs), `1` elite pair, `POX` ($p_c = 0.85$), `swap/insert` ($p_m = 0.15$), `20` frozen SHA-256 master seeds, and identical $B_{\text{GA}}$.

### Experiment A — Per-Individual LS Evaluation Cap (`Dev` Split Only)
Fix `f_LS = 1.00`, `steepest_descent + N5`, and compare:
- `DV_A1_B10`: $b_{\text{LS}} = 10$
- `DV_A2_B20`: $b_{\text{LS}} = 20$
- `DV_A3_B30`: $b_{\text{LS}} = 30$
- `DV_A4_B50`: $b_{\text{LS}} = 50$ *(Control — bit-for-bit identical to frozen `C06`)*

Select **$b_{\text{LS}}^*$** strictly on Dev by lowest mean instance-level median RPD (tie-breaks: Dev Friedman rank, anytime RPD AUC, mean runtime).

### Experiment B — Deterministic LS Candidate Selection (`Dev` Split Only)
Using frozen $b_{\text{LS}}^*$ and `steepest_descent + N5`, compare LS application fraction:
- `DV_B1_F020`: top $20\%$ ($K = \lceil 0.20 M \rceil$ pairs per cohort)
- `DV_B2_F040`: top $40\%$ ($K = \lceil 0.40 M \rceil$ pairs per cohort)
- `DV_B3_F060`: top $60\%$ ($K = \lceil 0.60 M \rceil$ pairs per cohort)
- `DV_B4_F100`: $100\%$ *(control carried from Experiment A at $b_{\text{LS}}^*$)*

**Deterministic Candidate Selection & Tie-Breaking:** Decode all $M$ pairs in the current cohort cleanly (`1` evaluation per genotype charged to `EvalCounter`), rank pairs ascending by best raw pair makespan key:
`key(i) = (min(raw_cmax_L, raw_cmax_B), max(raw_cmax_L, raw_cmax_B), pair_id)`,
and apply LS to the top $K$ pairs in rank order. Unselected pairs keep their validated raw makespans and genotypes (`0` LS evals). Rationale: prioritizes promising offspring without changing population size or adding extra surrogate evaluations. Select **$f_{\text{LS}}^*$** on Dev.

### Experiment C — LS Strategy Under Matched Fitness-Evaluation Budgets (`Dev` Split Only)
Using frozen $(b_{\text{LS}}^*, f_{\text{LS}}^*)$ and `N5`, compare:
- `DV_C1_SD`: Steepest Descent (`steepest_descent`, carried from Experiment B winner)
- `DV_C2_FI`: First Improvement (`first_improvement`, canonical deterministic move scan order)
- `DV_C3_SA`: Budget-Limited Simulated Annealing (`budget_limited_sa`: `n_cal = clamp(b_LS / 5, 2, 8)` probes retaining any improving move seen in calibration, $T_0 = \max(0.5, -\bar{\Delta}^+ / \ln(0.50))$, geometric cooling $\gamma = 0.85$ every `epoch_len = max(2, b_LS / 5)` probes, best visited schedule restored).

Select **$S_{\text{LS}}^*$** on Dev, freeze `DV_STAR = (b_LS^*, f_LS^*, S_LS^*)`, and lock before running Stage 4.

### Stage 4 — Frozen Held-Out Validation
Evaluate frozen `DV_STAR` and `DV_CTRL_C06` (`b_LS=50, f_LS=1.0, SD+N5`) on the `16` Held-Out instances (`20` seeds each), comparing against frozen `C06` and single-view `C02` (`LGA`, read from `results/phase_c/raw/phase_c_common_32.jsonl`) via blocked Friedman, Kendall $W$, paired Wilcoxon with Holm–Bonferroni correction, Vargha–Delaney $A_{12}$, Cliff's $\delta$, branch telemetry, generation counts, and diversity trajectories.

---

## 3. Matched Retuned Single-View Control (`C02_RETUNED_SA`)

To separate the effect of the retuned LS regime (`b_LS=10, f_LS=0.60, BudgetLimitedSa + N5`) from the Dual-View architecture itself (`DV_STAR`), a single final controlled experiment evaluates single-view Lamarckian GA (`C02_RETUNED_SA`, `C02` base) under identical capacity, initializer, seeds, and shared budget:

- **Genotype Capacity & Cohort Parity:** `N = 64` single-view Lamarckian genotypes, `E = 2` elites, `62` non-elite offspring per generation $t \ge 1$ (`POX` $p_c = 0.85$, swap/insert $p_m = 0.15$, `A11` init, `20` frozen SHA-256 seeds, identical $B_{\text{GA}}$).
- **Deterministic Candidate Selection (`f_LS = 0.60`):** Decode the $M$ newly generated chromosomes ($M = 64$ at $t=0$; $M = 62$ at $t \ge 1$) charging `1` decode eval each to `EvalCounter`, rank ascending by `(raw_makespan, cohort_idx)`, apply `BudgetLimitedSa` (`N5`, $b_{\text{LS}}=10$) and `apply_lamarckian_writeback` to the top $K = \mathrm{clamp}(\lceil f_{\text{LS}} \cdot M \rceil, 1, M)$ individuals ($K = 39$ at $t=0$, vs. `40` in `DV_STAR`; $K = 38$ at every $t \ge 1$, identical to `DV_STAR`), leaving unselected offspring at their raw makespans/genotypes (`0` LS evals).
- **Budget Conservation & Parity Controls:** Enforces `evals_used == init_evals + search_evals == decode_evals + ls_evals <= B_GA` with `bald_decode_evals == bald_ls_evals == 0`. Control `C02_CTRL_SD50` (`f_LS=1.00, b_LS=50, steepest_descent`) must match frozen `C02` bit-for-bit across all `320` Dev and `320` Held-Out runs. Dev evaluation is locked before Held-Out execution without parameter re-tuning.
