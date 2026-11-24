# Phase C Summary — Memetic & Dual-View Genetic Algorithms (`C01`–`C11`)

- **Transferred Frozen Components:** $A^* =$ `A11` (`Scrambled Sobol + MWR, alpha=1.0`), $B^* =$ `B21` (`Simulated Annealing (SA) + N5`)
- **Embedded Local Search (`C02`–`C11`, `docs/experiments/phase_c.md` §3):** `steepest_descent + N5` with $b_{\text{LS}} = 50$ evaluation cap per offspring decode *(intentionally distinct from full `B21` `Simulated Annealing + N5`, because `B21`'s 50-evaluation $T_0$ calibration would consume the entire `b_LS = 50` budget per call; full `B21` is preserved in `P3` and Stage 2 of `P5`)*.
- **Selected & Frozen Numerical Phase C Winner ($C^*$):** **`C02` — Lamarckian GA (LGA)** (`winner_lock_sha256`: `f6764fb58b76e6755ba2868a1974e1aec7a4dd8a9b46061a8bd18d6c616e90bc`)
- **Development Split (16 instances):** Mean instance-level median `RPD_BKS` = `7.570%`, Mean Best `RPD_BKS` = `5.701%`, Friedman rank = `2.56` / 11.
- **Held-Out Generalization (`C02`, 16 held-out instances):** Mean instance-level median `RPD_BKS` = `7.845%`, Mean Best `RPD_BKS` = `5.463%`, Friedman rank = `1.25` / 11, Hit BKS = `0.006`.
- **Experimental Volume & Blocked Friedman:** `7,040` runs (`32` instances $\times$ `11` configs $\times$ `20` frozen seeds) under equalized budget $B_{\text{GA}} = \mathrm{clamp}(64\lceil 2.5JM\rceil, 6400, 64000)$; Dev $\chi^2_F = 81.82$ ($p = 2.204 \times 10^{-13}$, Kendall $W = 0.5114$), Held-Out $\chi^2_F = 105.60$ ($p = 4.106 \times 10^{-18}$, Kendall $W = 0.6600$).

---

## 1. All 11 Phase C Configurations — Development & Held-Out Summary

Source CSVs: [`phase_c_dev_rankings.csv`](file:///home/abdelhalimyasser/Downloads/jssp-research/results/phase_c/analysis/phase_c_dev_rankings.csv), [`phase_c_held_out_evaluation.csv`](file:///home/abdelhalimyasser/Downloads/jssp-research/results/phase_c/analysis/phase_c_held_out_evaluation.csv).

| Rank (Dev) | ID | Method | Dev Med RPD | Dev Best RPD | Dev Rank | HO Med RPD | HO Best RPD | HO Rank | Dev / HO Gate $p_T$ | Dev / HO Final Div (`OBR` / $L$ / $B$) | Dev / HO Gens | Dev / HO ms |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|
| 1 | `C02` | Lamarckian GA (LGA) | `7.570%` | `5.701%` | `2.56` | `7.845%` | `5.463%` | `1.25` | `—` | `0.826` / `0.831` | `15.37` / `13.23` | `454.3` / `448.5` |
| 2 | `C01` | Classical GA Baseline | `7.823%` | `5.002%` | `5.62` | `8.548%` | `5.174%` | `5.72` | `—` | `0.580` / `0.506` | `474.06` / `491.81` | `833.6` / `810.5` |
| 3 | `C04` | Partial Lamarckian GA (PLGA 50%) | `8.299%` | `6.088%` | `3.81` | `8.849%` | `6.370%` | `3.47` | `—` | `0.845` / `0.862` | `13.66` / `12.07` | `609.3` / `570.0` |
| 4 | `C10` | Unpaired Dual-Pop Control (32 LGA + 32 BGA) | `8.353%` | `6.173%` | `4.19` | `8.844%` | `6.221%` | `3.59` | `—` | `0.867` (`0.805`/`0.810`) / `0.878` (`0.822`/`0.824`) | `13.41` / `12.02` | `572.1` / `541.0` |
| 5 | `C08` | Dual-View Fixed Gate (`p=0.75`) | `8.555%` | `6.133%` | `4.88` | `9.221%` | `6.283%` | `5.09` | `0.750` / `0.750` | `0.871` (`0.803`/`0.829`) / `0.885` (`0.829`/`0.851`) | `13.32` / `12.07` | `601.8` / `575.5` |
| 6 | `C09` | Dual-View Adaptive Ablation (`w_div=0.00`) | `8.735%` | `6.355%` | `6.25` | `9.601%` | `6.812%` | `6.69` | `0.515` / `0.523` | `0.874` (`0.820`/`0.827`) / `0.889` (`0.845`/`0.853`) | `13.22` / `11.92` | `556.8` / `556.6` |
| 7 | `C06` | Dual-View Adaptive Gate (`w_perf=0.70, w_div=0.30`) | `8.943%` | `6.674%` | `6.75` | `9.472%` | `6.695%` | `6.09` | `0.509` / `0.517` | `0.873` (`0.818`/`0.823`) / `0.888` (`0.843`/`0.847`) | `13.15` / `11.95` | `562.3` / `524.3` |
| 8 | `C11` | Dual-View Adaptive + Cross-View Migration (Exploratory) | `8.973%` | `6.900%` | `6.91` | `9.595%` | `6.935%` | `6.97` | `0.506` / `0.514` | `0.873` (`0.823`/`0.840`) / `0.887` (`0.847`/`0.857`) | `13.17` / `11.96` | `600.7` / `559.3` |
| 9 | `C05` | Dual-View Fixed Gate (`p=0.50`) | `9.027%` | `6.589%` | `7.09` | `9.740%` | `6.611%` | `7.12` | `0.500` / `0.500` | `0.875` (`0.820`/`0.836`) / `0.888` (`0.842`/`0.848`) | `13.19` / `11.93` | `551.0` / `561.4` |
| 10 | `C07` | Dual-View Fixed Gate (`p=0.25`) | `9.584%` | `7.317%` | `8.66` | `10.581%` | `7.593%` | `9.12` | `0.250` / `0.250` | `0.878` (`0.831`/`0.834`) / `0.893` (`0.862`/`0.851`) | `13.06` / `11.85` | `577.0` / `538.0` |
| 11 | `C03` | Baldwinian GA (BGA) | `10.318%` | `8.409%` | `9.28` | `11.701%` | `8.854%` | `10.88` | `—` | `0.873` / `0.868` | `11.95` / `11.12` | `554.5` / `540.4` |

---

## 2. Planned Holm–Wilcoxon Paired Contrasts (`C01`–`C11`)

Source: [`phase_c_statistical_tests.csv`](file:///home/abdelhalimyasser/Downloads/jssp-research/results/phase_c/analysis/phase_c_statistical_tests.csv). Negative `median_diff` favors `candidate`.

| Contrast Family | Candidate vs. Ref | Dev W/T/L | Dev $p_{\text{Holm}}$ (Sig?) | Held-Out W/T/L | Held-Out $p_{\text{raw}}$ | Held-Out $p_{\text{Holm}}$ (Sig?) | HO $A_{12}$ |
|---|---|---:|---|---:|---:|---|---:|
| `memetic_vs_classical` | `C02` vs `C01` | `9/4/3` | `0.479` (No) | `12/0/4` | `0.159` | `0.477` (No) | `0.570` |
| `memetic_vs_classical` | `C03` vs `C01` | `0/5/11` | `0.0167` (**Yes**, `C03` worse) | `1/0/15` | `1.526e-4` | `7.629e-4` (**Yes**, `C03` worse) | `0.340` |
| `memetic_vs_classical` | `C04` vs `C01` | `8/4/4` | `0.695` (No) | `10/0/6` | `0.782` | `0.782` (No) | `0.523` |
| `memetic_vs_classical` | `C05` vs `C01` | `4/5/7` | `0.301` (No) | `6/1/9` | `0.0995` | `0.398` (No) | `0.443` |
| `memetic_vs_classical` | `C06` vs `C01` | `5/4/7` | `0.408` (No) | `7/1/8` | `0.281` | `0.561` (No) | `0.463` |
| `dual_view_vs_single` | `C05` vs `C02` | `0/4/12` | `0.0133` (**Yes**, `C02` better) | `0/0/16` | `3.052e-5` | `1.831e-4` (**Yes**, `C02` better) | `0.383` |
| `dual_view_vs_single` | `C05` vs `C03` | `10/6/0` | `0.0133` (**Yes**, `C05` better) | `16/0/0` | `3.052e-5` | `1.831e-4` (**Yes**, `C05` better) | `0.602` |
| `dual_view_vs_single` | `C05` vs `C04` | `0/4/12` | `0.0133` (**Yes**, `C04` better) | `0/1/15` | `6.550e-4` | `1.310e-3` (**Yes**, `C04` better) | `0.436` |
| `dual_view_vs_single` | `C06` vs `C02` | `0/4/12` | `0.0133` (**Yes**, `C02` better) | `0/0/16` | `3.052e-5` | `1.831e-4` (**Yes**, `C02` better) | `0.391` |
| `dual_view_vs_single` | `C06` vs `C03` | `11/4/1` | `0.0133` (**Yes**, `C06` better) | `16/0/0` | `3.052e-5` | `1.831e-4` (**Yes**, `C06` better) | `0.621` |
| `dual_view_vs_single` | `C06` vs `C04` | `1/4/11` | `0.0133` (**Yes**, `C04` better) | `4/0/12` | `2.136e-3` | `2.136e-3` (**Yes**, `C04` better) | `0.457` |
| `adaptive_vs_fixed` | `C06` vs `C05` | `7/4/5` | `0.638` (No) | `11/0/5` | `0.0654` | `0.131` (No) | `0.516` |
| `adaptive_vs_fixed` | `C06` vs `C07` | `10/4/2` | `0.0142` (**Yes**, `C06` better) | `13/2/1` | `1.887e-3` | `5.662e-3` (**Yes**, `C06` better) | `0.570` |
| `adaptive_vs_fixed` | `C06` vs `C08` | `1/4/11` | `0.0193` (**Yes**, `C08` better) | `4/1/11` | `0.0884` | `0.131` (No) | `0.471` |
| `diversity_ablation` | `C06` vs `C09` | `5/4/7` | `0.347` (No) | `8/2/6` | `0.363` | `0.363` (No) | `0.504` |
| `paired_vs_unpaired` | `C05` vs `C10` | `1/4/11` | `5.747e-3` (**Yes**, `C10` better) | `2/0/14` | `3.052e-4` | `6.104e-4` (**Yes**, `C10` better) | `0.434` |
| `paired_vs_unpaired` | `C06` vs `C10` | `1/4/11` | `0.0121` (**Yes**, `C10` better) | `2/1/13` | `2.611e-3` | `2.611e-3` (**Yes**, `C10` better) | `0.455` |
| `exploratory_migr` | `C11` vs `C06` | `5/4/7` | `0.814` (No) | `5/2/9` | `0.300` | `0.300` (No) | `0.492` |

### Key Scientific Positioning

1. **Numerical Winner vs. Statistical Comparison (`C02` vs `C01`):** `C02` is the numerical winner on both splits (`7.570%` Dev / `7.845%` Held-Out Median RPD, Friedman rank `2.56` / `1.25`), yet `C02` vs `C01` did **not** achieve significance after Holm correction (`12/0/4`, $p_{\text{Holm}} = 0.477$ on Held-Out), and `C01` achieved lower mean Best RPD (`5.002%` Dev / `5.174%` Held-Out vs `5.701%` / `5.463%` for `C02`) by executing ~480 generations compared to `C02`'s ~14 generations.
2. **Honest Negative Result on Tested Dual-View Protocol:** Under the equalized $B_{\text{GA}}$ budget and Model A architecture, Dual-View variants (`C05`–`C09`, `C11`) did not outperform single-view `C02` (`16/0/0` losses on Held-Out for `C05` and `C06`, $p_{\text{Holm}} = 1.831 \times 10^{-4}$) or unpaired control `C10`.
3. **Within-Mechanism Structure Preserved:** Within Dual-View / Baldwinian configurations, higher Lamarckian emphasis improves performance monotonically (`C03` BGA `11.701%` $\to$ `C07` $p=0.25$ `10.581%` $\to$ `C05` $p=0.50$ `9.740%` $\to$ `C06` adaptive `9.472%` $\to$ `C08` $p=0.75$ `9.221%`), and `C06` significantly outperforms `C03` and `C07` on both splits. The diversity signal (`C06` vs `C09`, $p_{\text{Holm}} = 0.363$ Held-Out) and periodic migration (`C11` vs `C06`, $p_{\text{Holm}} = 0.300$ Held-Out) showed no statistically significant benefit under the tested budget.

---

## 3. Diagnostic Investigation: Why Dual-View Did Not Outperform LGA (`C02`)

Based strictly on [`crates/jssp-core/src/memetic/`](file:///home/abdelhalimyasser/Downloads/jssp-research/crates/jssp-core/src/memetic/dual_view/mod.rs) and saved telemetry across `7,040` raw runs (`results/phase_c/raw/phase_c_common_32.jsonl`):

1. **Effective Generations per View (Data-Supported):** Under equalized `B_GA` (`29,954` mean evaluations), embedded LS limits memetic runs to `13.06`–`15.37` generations (Dev) and `11.85`–`13.23` (Held-Out), compared to `474.06` / `491.81` in `C01`. Every Dual-View run still reaches global convergence too slowly (`best_found_generation` Held-Out `8.75`–`10.13` vs `10.98` in `C02`), leaving only ~`12` generations for evolutionary feedback.
2. **Objective-Evaluation Allocation Between Branches (Data-Supported):** `eval_dual_pair_views` unconditionally evaluates and applies LS (`b_LS=50`) to *both* $G_c^L$ and $G_c^B$ for every offspring pair. Thus `50%` of the evaluation budget is spent on the Baldwinian branch $G^B$, even though in `C05` and `C06` on Held-Out the Lamarckian branch finishes with strictly lower best makespan in `294/320` runs (`B` worse by `4.95%` on average) and `290/320` runs (`B` worse by `5.34%`), respectively.
3. **Pair-Level Selection Semantics (Data + Architecture):** Model A's weighted tournament $S_i(t) = p_t S_i^L + (1 - p_t) S_i^B$ mixes fitness across the two uncrossed lineages: unless $p_t \to 1$, individuals whose $G_i^L$ is weaker can win tournaments via $G_i^B$ (or vice versa), diluting selection pressure on the stronger $G^L$ lineage without redirecting evaluation budget toward it. Empirically, `C10` (unpaired independent tournaments within `32 LGA + 32 BGA`) significantly outperforms both paired `C05` and `C06` on Dev and Held-Out.
4. **Gate Convergence Speed & Dynamic Range (Data-Supported):** With $p_0 = 0.50$, $\lambda = 0.30$, $\eta = 0.05$, and `12`–`13` generations, the adaptive gate has a maximum theoretical drift of $\approx \pm 0.55$ per run if saturated, and in practice moves only marginally: on Held-Out, `C06` ends at $p_T = 0.5172 \pm 0.0185$ (range `[0.4701, 0.5830]`, mean within-run span `0.0197`), and `C09` ends at $0.5235 \pm 0.0251$ (`[0.4540, 0.6263]`, span `0.0268`). Thus `C06` stays near `C05` (`p=0.50`) rather than reaching `C08` (`p=0.75`).
5. **Genotype Diversity Retention Regime (Data-Supported):** The core hypothesis motivating Dual-View—that `C02` would suffer rapid genotypic diversity collapse requiring a Baldwinian reserve—did **not** trigger within ~`13`–`15` generations: `C02` still ended with `0.826` Dev / `0.831` Held-Out OBR Hamming diversity (vs `0.580` / `0.506` in `C01` at ~`480` generations). Consequently, `C06`'s diversity preservation (`0.843` / `0.847` view diversity) conferred no net advantage over `C02`.
6. **Cross-View Information Flow (Data-Supported):** In base Dual-View (`C05`–`C09`), crossover is strictly within-view, so improvements discovered in $G^L$ can never seed $G^B$ (and vice versa). Even in exploratory `C11` ($T_{\text{mig}} = 10$), with mean `11.96` Held-Out generations migration fired on average once (`1.70` non-duplicate transfers/run on Held-Out), too infrequently and too late to alter search outcomes (`C11` vs `C06` $p_{\text{Holm}} = 0.300$).
7. **Local Search Cost Asymmetry (Data-Supported):** Embedded `steepest_descent + N5` accounts for **`96.9%`–`97.5%`** of all evaluations across `C02`–`C11`. Because Baldwinian genotypes are not written back, post-crossover Baldwinian offspring start farther from local optima and consume **`35.4`–`38.6`** LS evaluations per decode on Held-Out compared to **`32.77`** in `C02` (`29.31` Dev vs `32.06`–`32.38` for Dual-View Dev).
8. **Effective Mating Pool Capacity & Elitism (Design + Data):** Maintaining $N_{\text{pairs}} = 32$ paired individuals to preserve the $N = 64$ genotype cap restricts the Lamarckian view to $32$ individuals and $E = 1$ elite pair (`max(1, elitism_count/2)`), halving the effective Lamarckian breeding pool compared to `C02`'s $N = 64, E = 2$.
9. **Runtime & Anytime Convergence Trade-off (Data-Supported):** `C02` is both faster (`454.3 ms` Dev / `448.5 ms` Held-Out vs `524.3`–`601.8 ms` for `C05`–`C11`) and dominates anytime convergence AUC (`10.270%` Dev / `11.461%` Held-Out mean raw `anytime_rpd_auc` vs `11.273%`–`13.318%`).
10. **Adaptive Gate Signal Opposition in `compute_delta_perf` (Architectural + Data):** In `AdaptiveGateState::compute_delta_perf`, $\Delta_{\text{perf}}(t)$ averages quality advantage $\bar{S}^L - \bar{S}^B$ with *single-step LS improvement yield* $(\bar{\delta}^L - \bar{\delta}^B)/(\bar{\delta}^L + \bar{\delta}^B)$. Because `C02` write-back (`G^L`) already starts children near local optima, marginal single-step LS improvement $\bar{\delta}^L$ shrinks relative to raw-genotype Baldwinian children (`G^B`), opposing $\bar{S}^L - \bar{S}^B > 0$ while $\Delta_{\text{div}} = D_L - D_B$ (`-0.0055` Dev / `-0.0042` Held-Out on `C06`) also pulls downward. Notice `C09` ($w_{\text{div}}=0$) drifted slightly higher ($p_T = 0.5235$) than `C06` ($0.5172$).

*Boundary between Data & Future Hypotheses:* Whether a Baldwinian reserve provides benefit under substantially larger evaluation budgets (e.g., $10\times$ higher $B_{\text{GA}}$ where `C02` experiences >`50` generations and genuine diversity loss), or when $p_t$ directly allocates LS/reproductive evaluations (adaptive Model B without breaking capacity accounting) rather than fixed 50/50 Model A evaluations, remains an **open hypothesis for future exploratory work**—not a conclusion inferred from aggregate Phase C results.

---

## Key Artifacts
- Frozen Winner Lock (`C02`): [`results/phase_c/winner/frozen_winner_c.json`](file:///home/abdelhalimyasser/Downloads/jssp-research/results/phase_c/winner/frozen_winner_c.json)
- Statistical Tests: [`results/phase_c/analysis/phase_c_statistical_tests.json`](file:///home/abdelhalimyasser/Downloads/jssp-research/results/phase_c/analysis/phase_c_statistical_tests.json), [`phase_c_statistical_tests.csv`](file:///home/abdelhalimyasser/Downloads/jssp-research/results/phase_c/analysis/phase_c_statistical_tests.csv)
- Summary Tables: [`phase_c_dev_rankings.csv`](file:///home/abdelhalimyasser/Downloads/jssp-research/results/phase_c/analysis/phase_c_dev_rankings.csv), [`phase_c_held_out_evaluation.csv`](file:///home/abdelhalimyasser/Downloads/jssp-research/results/phase_c/analysis/phase_c_held_out_evaluation.csv)
- Manifest: [`results/phase_c/manifests/phase_c_manifest.json`](file:///home/abdelhalimyasser/Downloads/jssp-research/results/phase_c/manifests/phase_c_manifest.json)
