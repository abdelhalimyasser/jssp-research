# Classical JSSP Research Framework — Final Research Results

## 1. Executive Summary

Across all three sequential research phases (`63` active experimental configurations: `A01`–`A24`, `B01`–`B24, B29`–`B32`, and `C01`–`C11`, with `B25`–`B28` excluded per `D04-A`) and the integrated pipeline validation (`P1`–`P5`), the strongest pipeline on both the **Development (`16` instances)** and held-out **Test (`16` instances)** splits is **`P3_FrozenA_FrozenB` (`A11` Scrambled Sobol + MWR ($\alpha = 1.0$) followed by `B21` Simulated Annealing + `N5`)**.

- **Frozen Phase A Winner ($A^*$):** `A11` — Scrambled Sobol + MWR ($\alpha = 1.0$) ([frozen_winner_a.json](file:///home/abdelhalimyasser/Downloads/jssp-research/results/phase_a/winner/frozen_winner_a.json))
- **Frozen Phase B Winner ($B^*$):** `B21` — Simulated Annealing (SA) + `N5` ([frozen_winner_b.json](file:///home/abdelhalimyasser/Downloads/jssp-research/results/phase_b/winner/frozen_winner_b.json))
- **Frozen Phase C Winner ($C^*$):** `C02` — Lamarckian GA (LGA) ([frozen_winner_c.json](file:///home/abdelhalimyasser/Downloads/jssp-research/results/phase_c/winner/frozen_winner_c.json))
- **Frozen Final Integrated Pipeline Winner:** **`P3_FrozenA_FrozenB`** ([frozen_final_pipeline.json](file:///home/abdelhalimyasser/Downloads/jssp-research/results/final_pipeline/winner/frozen_final_pipeline.json))

Under matched objective-evaluation budgets on the 16 held-out instances, `P3_FrozenA_FrozenB` achieves a mean Median RPD of **`3.959%`** and mean Best RPD of **`1.742%`** (Friedman rank `1.25 / 5`, mean runtime `532.0 ms`), outperforming the classical baseline `P1_Baseline_A01_C01` (`8.959%` Median RPD, `5.641%` Best RPD), `P2_FrozenA_C01` (`8.380%` Median RPD, `16/0/0` wins for `P3` vs `P2`, $p_{\text{Holm}} = 2.136 \times 10^{-4}$, $A_{12} = 0.863$), `P4_FrozenA_FrozenC` (`7.960%` Median RPD), and the three-stage hybrid `P5_Integrated_A_C_B` (`5.965%` Median RPD).

---

## 2. Experimental Protocol, Scale & Hardware Environment

All experiments target deterministic classical rectangular JSSP ($\min C_{\max}$) using the canonical Operation-Based Representation (OBR), deterministic active Serial SGS decoder (`D01-A`), and exact $O(L)$ disjunctive DAG evaluation (`D03-C`). Every phase strictly separated **Development (`16` instances)** for configuration selection from **Held-Out (`16` instances)** for confirmatory evaluation, with `20` frozen SHA-256 master seeds (`jssp-research-v1-sha256-20`, `ChaCha20Rng`).

| Benchmark / Execution Scope | Total Runs |
|---|---:|
| Phase A Layer A Standalone Initialization (`242` verified Track 1 instances) | `74,778` |
| Phase A Tier 2 Classical GA (`C01`, `32` Common Suite instances × `24` configs × `20` seeds) | `15,360` |
| Phase B Metaheuristics (`32` instances × `28` active configs × `20` seeds, budget $100L$) | `17,920` |
| Phase C Memetic GA (`32` instances × `11` configs × `20` seeds, budget $B_{\text{GA}}$) | `7,040` |
| Final Integrated Pipeline (`P1`–`P5`, `32` instances × `5` pipelines × `20` seeds) | `3,200` |
| **Total Recorded Runs** | **`118,298`** |

**Verified Execution Environment** ([final_pipeline_manifest.json](file:///home/abdelhalimyasser/Downloads/jssp-research/results/final_pipeline/manifests/final_pipeline_manifest.json)):
- **CPU:** `11th Gen Intel(R) Core(TM) i5-1145G7 @ 2.60GHz` (`4` physical / `8` logical cores)
- **Memory & OS:** `6872371200` bytes (~`6.4 GiB`) RAM, `Linux 7.0.0-43-generic x86_64`
- **Toolchain:** Python `3.15.2`, `uv 0.13.9`, `rustc 1.100.0`, `cargo 1.100.0`, `PyO3 0.30.1`, `maturin 1.16.0` (release build)

---

## 3. Sequential Phase Findings

### 3.1 Phase A — Initial Population Generation (`A01`–`A24`)
- **Winner Selection (Development):** **`A11` (`Scrambled Sobol + MWR (alpha=1.0)`)**, winner lock SHA-256 `90269dbd983236d8cbf0210e33fa470c6e1d4716597597d608a259456ab125c8` ([phase_a_summary.md](file:///home/abdelhalimyasser/Downloads/jssp-research/reports/phase_a_summary.md)).
- **Development (`16` instances):** `init_best_rpd_mean = 19.464%`, `init_obr_hamming_div_mean = 0.9252`, `ga_median_rpd_mean = 7.422%`, `ga_best_rpd_mean = 5.139%`, `friedman_ga_rank = 7.28 / 24`.
- **Held-Out (`16` instances):** `ga_median_rpd_mean = 8.252%`, `ga_best_rpd_mean = 5.325%`, `init_obr_hamming_div_mean = 0.9306`, `friedman_ga_rank = 6.84 / 24`.
- **Statistical Significance (Blocked Friedman on GA Median RPD):** Development $\chi^2_F = 153.25$ ($p = 3.132 \times 10^{-21}$, Kendall's $W = 0.4164$); Held-Out $\chi^2_F = 199.60$ ($p = 4.176 \times 10^{-30}$).
- **Structural Finding (QMC Scrambling Sensitivity):** Unscrambled Halton configurations (`A04`, `A08`, `A17`, `A18`) suffered severe initial diversity collapse (`0.561`–`0.626`) and degraded downstream GA performance (Dev GA RPD `16.47%`–`16.87%`), whereas Halton scrambling (`A05`: `7.982%`) and Scrambled Sobol + MWR combinations achieved high diversity and top downstream performance.

### 3.2 Phase B — Local Search & Metaheuristics (`B01`–`B24, B29`–`B32`)
- **Scope:** `28` active configurations evaluated from matched `A11` starting schedules under a shared $100L$ objective-evaluation budget; `B25`–`B28` (standalone Metropolis–Hastings) were removed prior to execution per `D04-A`. `rejected_cyclic_moves` remained `0.0` across all configurations ([phase_b_summary.md](file:///home/abdelhalimyasser/Downloads/jssp-research/reports/phase_b_summary.md)).
- **Winner Selection (Development):** **`B21` (`Simulated Annealing (SA) + N5`)**, winner lock SHA-256 `3951b6d9f41c2396b3fab2420e258d6f20b8677a6ae7402f41c85149b0350650`.
  - **Development (`16`):** `median_rpd_mean = 3.951%`, `best_rpd_mean = 1.975%`, `friedman_rank = 4.34 / 28`, `mean_elapsed_ms = 261.0`.
  - **Held-Out (`16`):** `median_rpd_mean = 4.349%`, `best_rpd_mean = 2.025%`, `improvement_pct_mean = 14.80%`, `friedman_rank = 2.88 / 28`.
- **Top Development Runners-Up (Median RPD):** `B22 (SA + N6)` `4.370%`, `B13 (Tabu + N5)` `4.471%`, `B23 (SA + N7)` `5.064%`, `B14 (Tabu + N6)` `5.502%`, `B30 (ILS-MH + N6)` `6.298%`, `B17 (ILS + N5)` `6.358%`.
- **Neighborhood Analysis:** Within standalone descents (`B01`–`B12`), `N8` led (`B12`/`B04`: `15.040%` vs `B01`: `16.013%`), while marginal Development Best RPD across all metaheuristics ranked `N6` (`9.738%`), `N5` (`9.913%`), `N7` (`10.173%`), `N8` (`11.194%`), with `SA + N5` (`B21`) winning overall.
- **Statistical Significance (Blocked Friedman):** Development $\chi^2_F = 342.60$ ($p = 2.119 \times 10^{-56}$, $W = 0.7931$); Held-Out $\chi^2_F = 396.47$ ($p = 2.610 \times 10^{-67}$).

### 3.3 Phase C — Memetic Genetic Algorithms (`C01`–`C11`)
- **Design & Budget Transfer Decision:** All Phase C variants initialized from `A11` under an equalized evaluation budget $B_{\text{GA}} = \mathrm{clamp}(64\lceil 2.5JM\rceil, 6400, 64000)$. Embedded local search in `C02`–`C11` used **`steepest_descent` + `N5`** at **$b_{\text{LS}} = 50$** (matching `docs/experiments/phase_c.md`, because `B21`'s SA 50-eval $T_0$ calibration would otherwise consume the entire $b_{\text{LS}} = 50$ budget per call), while full `B21` (`SA + N5`) was retained for `P3` and `P5`-Stage-2.
- **Winner Selection (Development):** **`C02` (`Lamarckian GA (LGA)`)**, winner lock SHA-256 `f6764fb58b76e6755ba2868a1974e1aec7a4dd8a9b46061a8bd18d6c616e90bc` ([phase_c_summary.md](file:///home/abdelhalimyasser/Downloads/jssp-research/reports/phase_c_summary.md)).

| Config | Variant | Dev Med RPD | Dev Best RPD | Dev Rank | Held-Out Med RPD | Held-Out Best RPD | Held-Out Rank |
|---|---|---:|---:|---:|---:|---:|---:|
| **`C02`** | **Lamarckian GA (LGA)** | **`7.570%`** | **`5.701%`** | **`2.56`** | **`7.845%`** | `5.463%` | **`1.25`** |
| `C01` | Classical GA | `7.823%` | — | `5.62` | `8.548%` | `5.174%` | `5.72` |
| `C04` | Partial Lamarckian GA (50%) | `8.299%` | — | `3.81` | `8.849%` | — | — |
| `C10` | Unpaired Dual-Population Control | `8.353%` | — | `4.19` | `8.844%` | — | — |
| `C08` | Dual-View Fixed Gate (`p=0.75`) | `8.555%` | — | `4.88` | `9.221%` | — | — |
| `C09` | Dual-View Adaptive (no diversity) | `8.735%` | — | `6.25` | `9.601%` | — | — |
| `C06` | Dual-View Adaptive Gate | `8.943%` | — | `6.75` | `9.472%` | — | — |
| `C11` | Dual-View + Migration | `8.973%` | — | `6.91` | `9.595%` | — | — |
| `C05` | Dual-View Fixed Gate (`p=0.50`) | `9.027%` | — | `7.09` | `9.740%` | — | — |
| `C07` | Dual-View Fixed Gate (`p=0.25`) | `9.584%` | — | `8.66` | `10.581%` | — | — |
| `C03` | Baldwinian GA (BGA) | `10.318%` | — | `9.28` | `11.701%` | — | — |

*(Mean Held-Out wall-clock time for `C02`: `448.5 ms`; `C06` Held-Out `mean gate p = 0.509`.)*

- **Statistical Significance (Blocked Friedman):** Development $\chi^2_F = 81.82$ ($p = 2.204 \times 10^{-13}$, $W = 0.5114$); Held-Out $\chi^2_F = 105.60$ ($p = 4.106 \times 10^{-18}$, $W = 0.6600$).
- **Honest Negative & Diagnostic Findings on Dual-View / BGA ([phase_c_statistical_tests.json](file:///home/abdelhalimyasser/Downloads/jssp-research/results/phase_c/analysis/phase_c_statistical_tests.json)):**
  - Under equalized total genotype and evaluation budgets ($B_{\text{GA}}$), **pure Baldwinian learning (`C03`) significantly underperforms Classical GA (`C01`)** on Held-Out (`1/0/15` wins/ties/losses, $p_{\text{Holm}} = 7.629 \times 10^{-4}$).
  - **Pure Lamarckian GA (`C02`) significantly outperforms both fixed (`C05`) and adaptive (`C06`) Dual-View variants** on every Held-Out instance (`0/0/16`, $p_{\text{Holm}} = 1.831 \times 10^{-4}$). Halving effective generational budget per view plus carrying a slower Baldwinian half outweighed the paired-diversity mechanism at this budget scale.
  - Within Dual-View variants on Held-Out, adaptive gating (`C06`) significantly outperforms pure Baldwinian (`C03`: `16/0/0`, $p_{\text{Holm}} = 1.831 \times 10^{-4}$) and `p=0.25` (`C07`: `13/2/1`, $p_{\text{Holm}} = 5.662 \times 10^{-3}$), while removing the diversity signal (`C06` vs `C09`, $p_{\text{Holm}} = 0.363$) and adding cross-view migration (`C11` vs `C06`, $p_{\text{Holm}} = 0.300$) showed no statistically significant Held-Out difference.

---

## 4. Final Integrated Pipeline Comparison (`P1`–`P5`)

Source data: [final_pipeline_manifest.json](file:///home/abdelhalimyasser/Downloads/jssp-research/results/final_pipeline/manifests/final_pipeline_manifest.json), [final_statistical_tests.json](file:///home/abdelhalimyasser/Downloads/jssp-research/results/final_pipeline/analysis/final_statistical_tests.json), [final_family_breakdown.csv](file:///home/abdelhalimyasser/Downloads/jssp-research/results/final_pipeline/analysis/final_family_breakdown.csv).

| Pipeline | Composition | Dev Med RPD | Dev Best RPD | Dev Hit BKS | Dev Rank | Dev Time (ms) | Held-Out Med RPD | Held-Out Best RPD | Held-Out Hit BKS | Held-Out Rank | Held-Out Time (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `P1_Baseline_A01_C01` | Random Init (`A01`) + Classical GA (`C01`) | `7.920%` | `5.116%` | — | `4.09` | — | `8.959%` | `5.641%` | — | `4.56` | `848.1` |
| `P2_FrozenA_C01` | Frozen $A^*$ (`A11`) + Classical GA (`C01`) | `7.885%` | `4.806%` | — | `3.72` | — | `8.380%` | `5.460%` | — | `3.94` | `847.7` |
| **`P3_FrozenA_FrozenB`** | **Frozen $A^*$ (`A11`) + Frozen $B^*$ (`B21` SA+`N5`)** | **`3.782%`** | **`2.156%`** | **`0.241`** | **`1.56`** | `544.4` | **`3.959%`** | **`1.742%`** | **`0.022`** | **`1.25`** | **`532.0`** |
| `P4_FrozenA_FrozenC` | Frozen $A^*$ (`A11`) + Frozen $C^*$ (`C02` LGA) | `7.888%` | `5.953%` | — | `3.38` | — | `7.960%` | `5.926%` | — | `3.09` | `509.3` |
| `P5_Integrated_A_C_B` | `A11` + `C02` (85% budget) + `B21` Polish (15%) | `5.834%` | `3.996%` | — | `2.25` | — | `5.965%` | `4.233%` | — | `2.16` | `549.4` |

### 4.1 Friedman & Holm–Wilcoxon Statistical Tests (`P1`–`P5`)

- **Blocked Friedman:**
  - **Development:** $\chi^2_F = 38.41$, $p = 9.222 \times 10^{-8}$, Kendall's $W = 0.6002$
  - **Held-Out:** $\chi^2_F = 45.61$, $p = 2.976 \times 10^{-9}$, Kendall's $W = 0.7126$
- **Held-Out Holm–Wilcoxon Contrasts (16 Held-Out instances):**
  - `P3_FrozenA_FrozenB` vs `P2_FrozenA_C01`: **`16/0/0`** ($p_{\text{Holm}} = 2.136 \times 10^{-4}$, $A_{12} = 0.863$, Cliff's $d = 0.727$)
  - `P5_Integrated_A_C_B` vs `P1_Baseline_A01_C01`: **`16/0/0`** ($p_{\text{Holm}} = 2.136 \times 10^{-4}$, $A_{12} = 0.684$)
  - `P5_Integrated_A_C_B` vs `P2_FrozenA_C01`: **`16/0/0`** ($p_{\text{Holm}} = 2.136 \times 10^{-4}$, $A_{12} = 0.664$)
  - `P5_Integrated_A_C_B` vs `P4_FrozenA_FrozenC`: **`11/1/4`** ($p_{\text{Holm}} = 0.0374$, $A_{12} = 0.592$)
  - `P5_Integrated_A_C_B` vs `P3_FrozenA_FrozenB`: **`2/0/14`** ($p_{\text{Holm}} = 1.709 \times 10^{-3}$, $A_{12} = 0.328$) — full-budget `P3` (`A11 + B21`) significantly outperforms splitting budget between Stage-1 LGA and Stage-2 SA polishing (`P5`).

### 4.2 Per-Family Best RPD for `P3_FrozenA_FrozenB` ([final_family_breakdown.csv](file:///home/abdelhalimyasser/Downloads/jssp-research/results/final_pipeline/analysis/final_family_breakdown.csv))

| Benchmark Family | `P3_FrozenA_FrozenB` Best RPD |
|---|---:|
| `LA` | `1.92%` |
| `ABZ` | `2.39%` |
| `FT` | `2.47%` |
| `ORB` | `3.30%` |
| `YN` | `4.30%` |
| `TA` | `4.44%` |
| `DMU` | `5.59%` |
| `SWV` | `9.59%` |

---

## 5. Artifact & Script Index

- Phase Reports: [phase_a_summary.md](file:///home/abdelhalimyasser/Downloads/jssp-research/reports/phase_a_summary.md), [phase_b_summary.md](file:///home/abdelhalimyasser/Downloads/jssp-research/reports/phase_b_summary.md), [phase_c_summary.md](file:///home/abdelhalimyasser/Downloads/jssp-research/reports/phase_c_summary.md)
- Frozen Winner Locks: [frozen_winner_a.json](file:///home/abdelhalimyasser/Downloads/jssp-research/results/phase_a/winner/frozen_winner_a.json), [frozen_winner_b.json](file:///home/abdelhalimyasser/Downloads/jssp-research/results/phase_b/winner/frozen_winner_b.json), [frozen_winner_c.json](file:///home/abdelhalimyasser/Downloads/jssp-research/results/phase_c/winner/frozen_winner_c.json), [frozen_final_pipeline.json](file:///home/abdelhalimyasser/Downloads/jssp-research/results/final_pipeline/winner/frozen_final_pipeline.json)
- Manifests & Analysis: [phase_a_manifest_index.json](file:///home/abdelhalimyasser/Downloads/jssp-research/results/phase_a/manifests/phase_a_manifest_index.json), [phase_b_manifest.json](file:///home/abdelhalimyasser/Downloads/jssp-research/results/phase_b/manifests/phase_b_manifest.json), [phase_c_manifest.json](file:///home/abdelhalimyasser/Downloads/jssp-research/results/phase_c/manifests/phase_c_manifest.json), [final_pipeline_manifest.json](file:///home/abdelhalimyasser/Downloads/jssp-research/results/final_pipeline/manifests/final_pipeline_manifest.json), [final_statistical_tests.json](file:///home/abdelhalimyasser/Downloads/jssp-research/results/final_pipeline/analysis/final_statistical_tests.json), [final_family_breakdown.csv](file:///home/abdelhalimyasser/Downloads/jssp-research/results/final_pipeline/analysis/final_family_breakdown.csv)
- Experiment Runners: [run_phase_a_pipeline.py](file:///home/abdelhalimyasser/Downloads/jssp-research/scripts/run_phase_a_pipeline.py), [run_phase_b_pipeline.py](file:///home/abdelhalimyasser/Downloads/jssp-research/scripts/run_phase_b_pipeline.py), [run_phase_c_pipeline.py](file:///home/abdelhalimyasser/Downloads/jssp-research/scripts/run_phase_c_pipeline.py), [run_final_pipeline.py](file:///home/abdelhalimyasser/Downloads/jssp-research/scripts/run_final_pipeline.py)
