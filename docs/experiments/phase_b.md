# Phase B Experiment Protocol — Local Search & Metaheuristics (`D03`, `D04`, `D05`)

**Status:** Reconciled Specification (**28 Active Configurations**: `B01`–`B24`, `B29`–`B32`; `B25`–`B28` pruned per `D04-A`; `B15-NNS` isolated supplementary ablation per `D03-B`)
**Algorithmic Specifications:** [`n5_n6_n7_n8_definitions.md`](file:///home/abdelhalimyasser/Downloads/jssp-research/docs/algorithms/n5_n6_n7_n8_definitions.md), [`mh_vs_sa.md`](file:///home/abdelhalimyasser/Downloads/jssp-research/docs/algorithms/mh_vs_sa.md)

---

## 1. Reconciled Configuration Matrix (28 Active + 1 Supplementary)

| Method | N5 | N6 | N7 | N8 | Status |
|---|:---:|:---:|:---:|:---:|---|
| **Steepest Descent (SD)** | `B01` | `B02` | `B03` | `B04` | Active |
| **First Improvement (FI)** | `B05` | `B06` | `B07` | `B08` | Active |
| **Random Improvement (RI)** | `B09` | `B10` | `B11` | `B12` | Active |
| **Tabu Search (TS)** | `B13` | `B14` | `B15` | `B16` | Active *(+ supplementary `B15-NNS`)* |
| **Iterated Local Search (ILS)** | `B17` | `B18` | `B19` | `B20` | Active |
| **Simulated Annealing (SA)** | `B21` | `B22` | `B23` | `B24` | Active |
| ~~**Standalone MH**~~ | ~~`B25`~~ | ~~`B26`~~ | ~~`B27`~~ | ~~`B28`~~ | **PRUNED / REJECTED (`D04-A`)** |
| **Metropolis-Filtered ILS Perturbation + LS** | `B29` | `B30` | `B31` | `B32` | Active (`D04-B`) |

---

## 2. Exact Strategy & Parameter Specifications (`D05`)

All move evaluations use **exact disjunctive-graph makespan evaluation** (`D03-C`), charging `1` unit to `eval_counter` per acyclic feasible neighbor evaluated.

1. **Canonical Deterministic Move Scan Order (shared by SD, FI, TS, ILS):**
   - Critical blocks ordered sequentially $B_1, \dots, B_b$ along the deterministic longest critical path $P$.
   - Within each block $B_l = (b_1, \dots, b_k)$, enumerate moves deterministically by `(move_kind, source_pos, target_pos, anchor_op_id)`, deduplicating identical resulting machine sequences while preserving first occurrence.
2. **Steepest Descent (`B01`–`B04`):**
   - Evaluates the full deduplicated feasible neighborhood $\mathcal{N}_k(\sigma)$ (unless `eval_counter == B_eval` mid-scan) and moves to $\arg\min_{\sigma' \in \mathcal{N}_k(\sigma)} C_{\max}(\sigma')$ if strictly improving ($C_{\max}(\sigma') < C_{\max}(\sigma)$), tie-breaking by canonical scan order.
   - **Budget-Exhaustion Rule:** Terminating mid-scan records `terminated_reason = BudgetExhausted` and **never** sets `reached_local_optimum = true`. Only an exhaustive scan with zero strictly improving moves sets `reached_local_optimum = true`.
3. **First Improvement (`B05`–`B08`):**
   - Iterates moves in canonical deterministic order and accepts immediately upon finding the first move with $C_{\max}(\sigma') < C_{\max}(\sigma)$. Stops at full-neighborhood scan without improvement (`reached_local_optimum = true`) or `BudgetExhausted`.
4. **Random Improvement (`B09`–`B12`):**
   - Draws a move uniformly **without replacement** using Fisher–Yates lazy permutation over $\mathcal{N}_k(\sigma)$ seeded by the `local_search_sample` PRNG substream, accepting the first strict improvement encountered. Avoids redundant re-evaluations within a step.
5. **Tabu Search (`B13`–`B16` and supplementary `B15-NNS`):**
   - *Tabu Attribute:* Reversed/reordered machine-pair arcs induced between moved operation $u$ and target anchor $v$ (following Zhang et al. 2007 / Xie et al. 2023 §4.3).
   - *Dynamic Tenure:* Drawn uniformly from $[L_{\min}, L_{\max}]$ per accepted move (`local_search_sample` stream), where $L_{\min} = 10 + \lfloor n / m \rfloor$ and $L_{\max} = \lfloor 1.5 L_{\min} \rfloor$.
   - *Aspiration Criterion:* Overrides tabu status iff $C_{\max}(\sigma') < C_{\max}(\sigma_{\text{best}})$.
   - *Move Selection & Diversification:* Selects best non-tabu/aspirated neighbor in $\mathcal{N}_k(\sigma)$; if all moves are tabu and non-aspirated, selects the oldest-tabu move; if $\sigma_{\text{best}}$ fails to improve for `improve_iter = 200` consecutive iterations, executes a single 3-cycle distinct-job OBR perturbation (`perturbation` stream, identical kernel to ILS) and resets the stagnation counter.
6. **Iterated Local Search (`B17`–`B20`):**
   - *Embedded Local Search:* **Steepest Descent (`B01`–`B04`)** across all four neighborhoods (all inner move evaluations charged to the shared run `eval_counter`).
   - *Perturbation Kernel:* Single **3-cycle distinct-job OBR permutation** (`Option B`: uniformly sample 3 positions $\{i_1 < i_2 < i_3\}$ with pairwise distinct job IDs and apply orientation $\sigma_+$ or $\sigma_-$ on $\pi_{\text{canon}}(S)$ via `perturbation` stream, degree $D_{3\text{cycle}} = 2\sum_{a<b<c} m_a m_b m_c$; structurally preserves OBR validity and Serial SGS schedule acyclicity).
   - *Acceptance Criterion:* Accept improved-or-equal local optimum ($C_{\max}(S_{\text{next}}) \le C_{\max}(S_{\text{curr}})$), with best-so-far $S_{\text{best}}$ always preserved independently.
   - *Restart Rule:* If `stagnation_ils = 20` perturbations yield no improvement to $S_{\text{best}}$, perform a stronger $k_{\text{restart}} = \max(2, \lfloor L / 15 \rfloor)$ sequential 3-cycle OBR perturbations from $S_{\text{best}}$.
7. **Simulated Annealing (`B21`–`B24`):**
   - *Proposal:* Uniform single-move sampling from $\mathcal{N}_k(\sigma)$ (`sa_accept` / `local_search_sample` streams).
   - *Cooling Schedule:* $T_0$ calibrated from 50 initial moves targeting $80\%$ acceptance ($T_0 = \max(1.0, -\overline{\Delta C^+}/\ln(0.80))$); geometric cooling $T_{t+1} = \max(10^{-3}, 0.95\, T_t)$ every epoch of $2L$ proposals.
8. **Metropolis-Filtered ILS Perturbation + Deterministic Local Search (`B29`–`B32`):**
   - Described accurately as a **Metropolis-filtered ILS perturbation followed by deterministic Steepest Descent local search** (never claiming composite MH detailed balance). Operates on OBR state $\pi = \pi_{\text{canon}}(S)$ with energy $E(\pi) = C_{\max}(\mathrm{Decode}(\pi))$ using the **3-cycle distinct-job OBR perturbation kernel** (`perturbation` stream, exact symmetric OBR degree $D_{3\text{cycle}} = 2\sum_{a<b<c} m_a m_b m_c$), decodes $S_{\text{pert}} = \mathrm{Decode}(\pi')$ (`1` eval), applies $\alpha_{\text{pert}}(\pi, \pi') = \min(1, \exp(-\beta_{\text{filter}}[E(\pi') - E(\pi)]))$ (`mh_filter` stream, $\beta_{\text{filter}} = 2.0 / \overline{p}$) before invoking Steepest Descent, and skips descent when rejected. See [`mh_vs_sa.md`](file:///home/abdelhalimyasser/Downloads/jssp-research/docs/algorithms/mh_vs_sa.md) §4 for OBR-space symmetry vs. quotient schedule-space asymmetry analysis.

---

## 3. Initial Solutions, Budgets & Supplementary Comparisons
- **Matched Starting Solutions:** Every configuration evaluates from identical paired initial schedules (`instance_id, seed_idx`), generated via `A01` Uniform Random OBR (`init` substream) so all methods start from identical starting schedules per `(instance, seed)`.
- **Budgets:**
  - *Development / Screening:* $B_{\text{eval}} = 100 \cdot L$ objective evaluations.
  - *Frozen Confirmatory:* $B_{\text{eval}} = 500 \cdot L$ objective evaluations.
- **Reported Telemetry:** Best/final $C_{\max}$, RPD, wall-clock seconds, `eval_counter`, `generated_moves`, `feasible_moves`, `rejected_cyclic_moves`, `clipped_moves`, `improving_moves`, `accepted_moves`, `adapter_left_shift_count`, `terminated_reason`, `reached_local_optimum`, and anytime $C_{\max}$-vs-evaluations trace.

### 3.1 Supplementary `NNS` Comparison Protocol (`B15-NNS` vs. `B15`)
- **Verified Comparator:** `B15` is **`Tabu Search + N7`** (row 4, column `N7` of Section 1).
- **Isolation Rule:** `B15-NNS` (`Tabu Search + NNS`, Gui et al., 2023) is executed strictly as a supplementary ablation outside the 28 active primary Phase B configurations.
- **Strict Pairing:** `B15-NNS` and `B15` must run on identical benchmark instances using **identical paired initial solutions (`A01` per `(instance, seed)`), identical objective-evaluation budgets ($100L$ screening / $500L$ confirmatory), identical 20 master seeds and named PRNG substreams, and identical Tabu Search parameters**.
