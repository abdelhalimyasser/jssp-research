# Dual-View Local Search Budget Investigation (Exploratory Extension)

> **Scope Note:** All experiments reported here are exploratory additions on branch `exploratory/dual-view-ls-budget` (`3,520` new runs). Frozen Phase A–C (`A11`, `B21`, `C02`) and Integrated Pipeline (`P3`) confirmatory outputs remain completely unmodified.

## 1. Implementation Audit Summary

1. **Strict Budget Accounting:** `EvalCounter` enforces `evals_used == init_evals + search_evals == decode_evals + ls_evals <= budget`. Control parity checks confirmed `DV_A4_B50` (Dev, `320` runs) and `DV_CTRL_C06` (Held-Out, `320` runs) match frozen `C06` bit-for-bit across makespans, evaluation counts, generations, and final gate weights.
2. **Why `96.9%`–`97.5%` Was Consumed by LS:** Calling `steepest_descent + N5` (`b_LS=50`) on 100% of both views consumes `~33.9` LS evaluations per single genotype decode (`97.1%` per-offspring cost), shrinking evolution to `~12` Held-Out generations and duplicating deterministic LS on identical $t=0$ genotypes across both views.
3. **Standalone SA Incompatibility Resolved:** Phase B `Simulated Annealing` probes 50 uncommitted moves for $T_0$ calibration, leaving 0 committed moves at $b_{\text{LS}} \le 50$. Experiment C evaluates `run_budget_limited_sa` (`n_cal = clamp(b_LS/5, 2, 8)` retaining best calibration moves).

## 2. Controlled Screening on Development Split (`16` Instances × `20` Seeds)

### Experiment A — Per-Individual LS Evaluation Cap (`f_LS = 1.00`, Steepest Descent + `N5`)

| ID | Strategy | $b_{\text{LS}}$ | $f_{\text{LS}}$ | Med RPD | Mean RPD | Best RPD | Friedman Rank | Gens | LS Share | Final OBR Div | ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `DV_A1_B10` | `steepest_descent` | `10` | `1.00` | `7.558%` | `7.587%` | `5.117%` | `1.81` | `45.38` | `89.5%` | `0.820` | `749.8` |
| `DV_A2_B20` | `steepest_descent` | `20` | `1.00` | `8.125%` | `8.104%` | `5.780%` | `2.38` | `25.16` | `93.7%` | `0.852` | `621.3` |
| `DV_A3_B30` | `steepest_descent` | `30` | `1.00` | `8.312%` | `8.412%` | `6.072%` | `2.50` | `18.29` | `95.0%` | `0.866` | `555.7` |
| `DV_A4_B50` | `steepest_descent` | `50` | `1.00` | `8.943%` | `8.907%` | `6.674%` | `3.31` | `13.15` | `95.9%` | `0.873` | `491.7` |

- **Dev Friedman:** $\chi^2_F = 14.70$ ($p = 2.092e-03$, Kendall $W = 0.3062$). Selected: **`DV_A1_B10`** ($b_{\text{LS}}^* = 10$).

| Candidate vs. Ref | W/T/L | Mean Diff | $p_{\text{raw}}$ | $p_{\text{Holm}}$ | Sig (0.05) | $A_{12}$ | Cliff's $d$ |
|---|---:|---:|---:|---:|---|---:|---:|
| `DV_A1_B10` vs `DV_A4_B50` | `11/4/1` | `-1.384%` | `2.873e-03` | `8.620e-03` | Yes | `0.547` | `+0.094` |
| `DV_A2_B20` vs `DV_A4_B50` | `9/4/3` | `-0.818%` | `1.206e-02` | `1.206e-02` | Yes | `0.531` | `+0.062` |
| `DV_A3_B30` vs `DV_A4_B50` | `11/4/1` | `-0.631%` | `4.742e-03` | `9.484e-03` | Yes | `0.543` | `+0.086` |

### Experiment B — Candidate Selection Fraction ($b_{\text{LS}}^* = 10$, Steepest Descent + `N5`)

| ID | Strategy | $b_{\text{LS}}$ | $f_{\text{LS}}$ | Med RPD | Mean RPD | Best RPD | Friedman Rank | Gens | LS Share | Final OBR Div | ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `DV_B3_F060` | `steepest_descent` | `10` | `0.60` | `6.745%` | `6.808%` | `4.674%` | `2.12` | `75.64` | `81.6%` | `0.777` | `1005.8` |
| `DV_B2_F040` | `steepest_descent` | `10` | `0.40` | `6.941%` | `6.949%` | `4.471%` | `2.34` | `108.00` | `73.6%` | `0.751` | `1017.3` |
| `DV_B1_F020` | `steepest_descent` | `10` | `0.20` | `6.977%` | `7.000%` | `4.726%` | `2.53` | `174.63` | `57.7%` | `0.736` | `1138.6` |
| `DV_B4_F100` | `steepest_descent` | `10` | `1.00` | `7.558%` | `7.587%` | `5.117%` | `3.00` | `45.38` | `89.5%` | `0.820` | `749.8` |

- **Dev Friedman:** $\chi^2_F = 5.46$ ($p = 1.410e-01$, Kendall $W = 0.1138$). Selected: **`DV_B3_F060`** ($f_{\text{LS}}^* = 0.60$).

| Candidate vs. Ref | W/T/L | Mean Diff | $p_{\text{raw}}$ | $p_{\text{Holm}}$ | Sig (0.05) | $A_{12}$ | Cliff's $d$ |
|---|---:|---:|---:|---:|---|---:|---:|
| `DV_B1_F020` vs `DV_B4_F100` | `9/4/3` | `-0.582%` | `5.974e-02` | `8.278e-02` | No | `0.523` | `+0.047` |
| `DV_B2_F040` vs `DV_B4_F100` | `8/4/4` | `-0.617%` | `4.139e-02` | `8.278e-02` | No | `0.520` | `+0.039` |
| `DV_B3_F060` vs `DV_B4_F100` | `9/4/3` | `-0.813%` | `1.502e-02` | `4.507e-02` | Yes | `0.527` | `+0.055` |

### Experiment C — LS Strategy Comparison ($b_{\text{LS}}^* = 10$, $f_{\text{LS}}^* = 0.60$, `N5`)

| ID | Strategy | $b_{\text{LS}}$ | $f_{\text{LS}}$ | Med RPD | Mean RPD | Best RPD | Friedman Rank | Gens | LS Share | Final OBR Div | ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `DV_C3_SA` | `budget_limited_sa` | `10` | `0.60` | `5.800%` | `5.846%` | `3.556%` | `1.44` | `71.13` | `83.4%` | `0.801` | `1217.0` |
| `DV_C2_FI` | `first_improvement` | `10` | `0.60` | `6.175%` | `6.231%` | `3.659%` | `1.94` | `77.79` | `80.6%` | `0.782` | `1138.5` |
| `DV_C1_SD` | `steepest_descent` | `10` | `0.60` | `6.745%` | `6.808%` | `4.674%` | `2.62` | `75.64` | `81.6%` | `0.777` | `1005.8` |

- **Dev Friedman:** $\chi^2_F = 15.17$ ($p = 5.089e-04$, Kendall $W = 0.4740$). Frozen Final Choice: **`DV_STAR` = `DV_C3_SA`** (`budget_limited_sa`, $b_{\text{LS}}^* = 10$, $f_{\text{LS}}^* = 0.60$).

| Candidate vs. Ref | W/T/L | Mean Diff | $p_{\text{raw}}$ | $p_{\text{Holm}}$ | Sig (0.05) | $A_{12}$ | Cliff's $d$ |
|---|---:|---:|---:|---:|---|---:|---:|
| `DV_C2_FI` vs `DV_C1_SD` | `10/4/2` | `-0.570%` | `6.040e-03` | `6.040e-03` | Yes | `0.539` | `+0.078` |
| `DV_C3_SA` vs `DV_C1_SD` | `12/4/0` | `-0.946%` | `2.218e-03` | `4.435e-03` | Yes | `0.551` | `+0.102` |
| `DV_C3_SA` vs `DV_C2_FI` | `9/4/3` | `-0.375%` | `2.806e-02` | `2.806e-02` | Yes | `0.527` | `+0.055` |

## 3. Held-Out Validation: Frozen `DV_STAR` vs. `C06` Control & Single-View `C02` (LGA)

### Held-Out Summary (`16` Held-Out Instances × `20` Seeds)

| ID | Strategy | $b_{\text{LS}}$ | $f_{\text{LS}}$ | Med RPD | Mean RPD | Best RPD | Friedman Rank | Gens | LS Share | Final OBR Div | ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `DV_STAR` | `budget_limited_sa` | `10` | `0.60` | `6.054%` | `6.080%` | `3.648%` | `1.38` | `68.62` | `85.8%` | `0.802` | `897.2` |
| `C02_LGA` | `Steepest Descent` | `50` | `1.00` | `7.845%` | `7.810%` | `5.463%` | `1.75` | `13.23` | `96.6%` | `0.831` | `448.5` |
| `DV_CTRL_C06` | `steepest_descent` | `50` | `1.00` | `9.472%` | `9.499%` | `6.695%` | `2.88` | `11.95` | `97.0%` | `0.888` | `473.4` |

- **Held-Out Friedman (`DV_STAR`, `DV_CTRL_C06`, `C02_LGA`):** $\chi^2_F = 19.50$ ($p = 5.829e-05$, Kendall $W = 0.6094$).

| Candidate vs. Ref | W/T/L | Mean Diff | $p_{\text{raw}}$ | $p_{\text{Holm}}$ | Sig (0.05) | $A_{12}$ | Cliff's $d$ |
|---|---:|---:|---:|---:|---|---:|---:|
| `DV_STAR` vs `DV_CTRL_C06` | `14/0/2` | `-3.419%` | `1.526e-04` | `1.526e-04` | Yes | `0.688` | `+0.375` |
| `DV_STAR` vs `C02_LGA` | `12/0/4` | `-1.791%` | `4.181e-03` | `4.181e-03` | Yes | `0.582` | `+0.164` |
| `DV_CTRL_C06` vs `C02_LGA` | `0/0/16` | `+1.627%` | `3.052e-05` | `3.052e-05` | Yes | `0.391` | `-0.219` |

### Branch-Specific Budget & Move Telemetry (`Stage 4 Held-Out`)

| ID | Branch | Decode Evals | LS Evals | LS Calls | Improving Calls | Accepted Moves | Rejected Moves | Cyclic Rejects |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `DV_STAR` | Lamarckian ($G^L$) | `2158.1` | `13095.8` | `1309.9` | `905.3` | `7072.8` | `6022.9` | `0.00` |
| `DV_STAR` | Baldwinian ($G^B$) | `2158.1` | `13092.0` | `1309.2` | `1241.9` | `7765.9` | `5326.1` | `0.00` |
| `DV_CTRL_C06` | Lamarckian ($G^L$) | `389.1` | `13894.8` | `389.1` | `342.9` | `1515.9` | `12378.9` | `0.00` |
| `DV_CTRL_C06` | Baldwinian ($G^B$) | `388.7` | `15831.3` | `388.7` | `383.4` | `1877.0` | `13954.3` | `0.00` |

### Held-Out Diversity Trajectories (`DV_STAR` vs `DV_CTRL_C06`)

| Experiment | Generation | Active Runs | Mean OBR Div | Mean $G^L$ Div | Mean $G^B$ Div |
|---|---:|---:|---:|---:|---:|
| `DV_STAR` | `0` | `320` | `0.9245` | `0.9229` | `0.9374` |
| `DV_STAR` | `5` | `320` | `0.9034` | `0.8849` | `0.8750` |
| `DV_STAR` | `10` | `320` | `0.8926` | `0.8562` | `0.8602` |
| `DV_STAR` | `15` | `320` | `0.8805` | `0.8229` | `0.8453` |
| `DV_STAR` | `20` | `300` | `0.8724` | `0.7991` | `0.8325` |
| `DV_STAR` | `30` | `300` | `0.8528` | `0.7448` | `0.8084` |
| `DV_STAR` | `50` | `180` | `0.8548` | `0.7316` | `0.7967` |
| `DV_STAR` | `144` | `20` | `0.8129` | `0.6258` | `0.7146` |
| `DV_STAR` | `final` | `320` | `0.8016` | `0.6184` | `0.7307` |
| `DV_CTRL_C06` | `0` | `320` | `0.9252` | `0.9111` | `0.9374` |
| `DV_CTRL_C06` | `5` | `320` | `0.9020` | `0.8799` | `0.8747` |
| `DV_CTRL_C06` | `10` | `180` | `0.9127` | `0.8840` | `0.8777` |
| `DV_CTRL_C06` | `15` | `101` | `0.9172` | `0.8900` | `0.8780` |
| `DV_CTRL_C06` | `20` | `20` | `0.9166` | `0.8797` | `0.8861` |
| `DV_CTRL_C06` | `22` | `6` | `0.8905` | `0.8248` | `0.8134` |
| `DV_CTRL_C06` | `final` | `320` | `0.8876` | `0.8432` | `0.8475` |

### Development Reference (`DV_STAR` vs. `DV_CTRL_C06` & `C02_LGA`)

| ID | Strategy | $b_{\text{LS}}$ | $f_{\text{LS}}$ | Med RPD | Mean RPD | Best RPD | Friedman Rank | Gens | LS Share | Final OBR Div | ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `DV_STAR` | `budget_limited_sa` | `10` | `0.60` | `5.800%` | `5.846%` | `3.556%` | `1.38` | `71.13` | `83.4%` | `0.801` | `1217.0` |
| `C02_LGA` | `Steepest Descent` | `50` | `1.00` | `7.570%` | `7.584%` | `5.701%` | `1.88` | `15.37` | `94.6%` | `0.826` | `454.3` |
| `DV_CTRL_C06` | `steepest_descent` | `50` | `1.00` | `8.943%` | `8.907%` | `6.674%` | `2.75` | `13.15` | `95.9%` | `0.873` | `491.7` |

| Candidate vs. Ref | W/T/L | Mean Diff | $p_{\text{raw}}$ | $p_{\text{Holm}}$ | Sig (0.05) | $A_{12}$ | Cliff's $d$ |
|---|---:|---:|---:|---:|---|---:|---:|
| `DV_STAR` vs `DV_CTRL_C06` | `12/4/0` | `-3.143%` | `2.218e-03` | `2.218e-03` | Yes | `0.594` | `+0.188` |
| `DV_STAR` vs `C02_LGA` | `10/4/2` | `-1.771%` | `6.040e-03` | `6.040e-03` | Yes | `0.559` | `+0.117` |
| `DV_CTRL_C06` vs `C02_LGA` | `0/4/12` | `+1.372%` | `2.218e-03` | `2.218e-03` | Yes | `0.434` | `-0.133` |

## 4. Evidence-Based Diagnostic Conclusions

1. **Was excessive LS consumption the primary cause of C06 underperformance?**
   **Yes — quantitatively confirmed across both Dev and Held-Out splits.** Under matched total budget, reducing only the per-individual LS cap (`b_LS: 50 -> 10` at `f_LS=1.00`, steepest descent) on Dev improved median RPD monotonically (`8.943% -> 8.312% -> 8.125% -> 7.558%`, `-1.384%`, `11/4/1`, `p_Holm = 8.620e-03`) and immediately reached parity with Dev `C02_LGA` (`7.570%`). Adding selective Top-60% screening (`f_LS* = 0.60`, `-0.813%`, `p_Holm = 4.507e-02`) and switching to `BudgetLimitedSa` (`-0.946%` vs `DV_C1_SD`, `12/4/0`, `p_Holm = 4.435e-03`) yielded **`DV_STAR`**, which cut Held-Out median RPD from `9.472%` (`DV_CTRL_C06`) to **`6.054%`** (`-3.419%`, `14/0/2`, `p_Holm = 1.526e-04`, $A_{12} = 0.688$).

2. **Did reallocating LS budget improve evolutionary horizon, gate adaptation, and search behavior?**
   **Yes.** On Held-Out, `DV_STAR` increased executed generations `5.74x` (`11.95 -> 68.62`) while lowering LS budget share from `97.0%` to `85.8%` (raising genotype decodes per branch from `~389` to `2,158.1`). This longer horizon allowed the adaptive gate ($G^B$ weight $p$) to respond properly instead of stalling near its initialization (`final_gate_p_mean = 0.3540` vs `0.5172`), shifting reproductive slots toward the stronger Lamarckian branch (`G^L` beat `G^B` in `302/320` = `94.4%` Held-Out runs) and allowing evolutionary selection/recombination to drive `final_obr_diversity` down productively (`0.9245 -> 0.8016`) rather than stalling near initial random diversity (`0.9252 -> 0.8876`).

3. **Did reallocated Dual-View close the gap to single-view LGA (`C02`)?**
   **It outperformed frozen `C02_LGA` under equal total evaluation budget, with an important experimental caveat.** On Held-Out, frozen `DV_STAR` outperformed frozen `C02_LGA` (`Med RPD 6.054%` vs `7.845%`, `-1.791%`, `12/0/4`, `p_Holm = 4.181e-03`, $A_{12} = 0.582$; `Best RPD 3.648%` vs `5.463%`). However, `C02_LGA` used the frozen Phase C confirmatory setting (`b_LS=50, f_LS=1.00, steepest_descent`, `13.23` gens). Whether `Dual-View` at `(b_LS=10, f_LS=0.60, budget_limited_sa)` outperforms an equivalently re-tuned single-view LGA (`C02` with `b_LS=10, f_LS=0.60, budget_limited_sa`) was outside this investigation's frozen-baseline comparison and remains an open ablation. Note also that under matched total evaluations (`30,504`), `DV_STAR` took more wall-clock time than `DV_CTRL_C06` (`897.2 ms` vs `473.4 ms`) due to executing `~5.7x` more generations of M2 serial decodes and crossover/diversity passes.

4. **Remaining architectural bottleneck in `Dual-View`:**
   Even in `DV_STAR`, symmetrical per-branch LS application still spends `50.0%` of all LS evaluations (`13,092.0 / 26,187.8`) refining Baldwinian genotypes (`G^B`) whose moves are never written back, while `G^L` produces the strictly better schedule in `94.4%` of Held-Out runs. Gate-coupled asymmetric LS allocation (allocating LS effort proportionally to $1-p$ vs $p$) or avoiding redundant $t=0$ LS on identical paired chromosomes is a clear next-step hypothesis.

## 5. Artifacts
- Protocol: [`docs/experiments/dual_view_ls_budget_protocol.md`](file:///home/abdelhalimyasser/Downloads/jssp-research/docs/experiments/dual_view_ls_budget_protocol.md)
- Selection Lock (`bf8cd2a2458254e3...`): [`results/exploratory_dv_ls/selection/frozen_dv_ls_selection.json`](file:///home/abdelhalimyasser/Downloads/jssp-research/results/exploratory_dv_ls/selection/frozen_dv_ls_selection.json)
- Manifest (`7109b6d761db5053...`): [`results/exploratory_dv_ls/manifests/exploratory_dv_ls_manifest.json`](file:///home/abdelhalimyasser/Downloads/jssp-research/results/exploratory_dv_ls/manifests/exploratory_dv_ls_manifest.json)
- Raw Runs JSONL (`3,520` runs): [`results/exploratory_dv_ls/raw/exploratory_dv_ls_runs.jsonl`](file:///home/abdelhalimyasser/Downloads/jssp-research/results/exploratory_dv_ls/raw/exploratory_dv_ls_runs.jsonl)
- Statistical Tests: [`results/exploratory_dv_ls/analysis/exploratory_statistical_tests.json`](file:///home/abdelhalimyasser/Downloads/jssp-research/results/exploratory_dv_ls/analysis/exploratory_statistical_tests.json)
