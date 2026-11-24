# Classical JSSP Research Framework — Final Research Results

## 1. Executive Summary

Across all three sequential research phases (`63` active experimental configurations: `A01`–`A24`, `B01`–`B24, B29`–`B32`, and `C01`–`C11`, with `B25`–`B28` excluded per `D04-A`) and the integrated pipeline validation (`P1`–`P5`), the strongest tested pipeline on both the **Development (`16` instances)** and held-out **Test (`16` instances)** splits is **`P3_FrozenA_FrozenB` (`A11` Scrambled Sobol + MWR ($\alpha = 1.0$) followed by `B21` Simulated Annealing + `N5`)**.

- **Frozen Phase A Winner ($A^*$):** `A11` — Scrambled Sobol + MWR ($\alpha = 1.0$) ([frozen_winner_a.json](file:///home/abdelhalimyasser/Downloads/jssp-research/results/phase_a/winner/frozen_winner_a.json))
- **Frozen Phase B Winner ($B^*$):** `B21` — Simulated Annealing (SA) + `N5` ([frozen_winner_b.json](file:///home/abdelhalimyasser/Downloads/jssp-research/results/phase_b/winner/frozen_winner_b.json))
- **Frozen Phase C Winner ($C^*$):** `C02` — Lamarckian GA (LGA) ([frozen_winner_c.json](file:///home/abdelhalimyasser/Downloads/jssp-research/results/phase_c/winner/frozen_winner_c.json)) *(numerical Phase C winner; see §3.3 for Holm–Wilcoxon comparisons)*
- **Frozen Final Integrated Pipeline Winner:** **`P3_FrozenA_FrozenB`** ([frozen_final_pipeline.json](file:///home/abdelhalimyasser/Downloads/jssp-research/results/final_pipeline/winner/frozen_final_pipeline.json))

Under matched objective-evaluation budgets on the 16 held-out instances, `P3_FrozenA_FrozenB` achieves a mean Median RPD of **`3.959%`** and mean Best RPD of **`1.742%`** (Friedman rank `1.25 / 5`, BKS hit rate `0.022`, mean runtime `532.0 ms`), significantly outperforming the classical baseline `P1_Baseline_A01_C01` (`8.959%` Median RPD, `5.641%` Best RPD), `P2_FrozenA_C01` (`8.380%` Median RPD, `16/0/0` wins for `P3` vs `P2`, $p_{\text{Holm}} = 2.136 \times 10^{-4}$, $A_{12} = 0.863$), `P4_FrozenA_FrozenC` (`7.960%` Median RPD), and the tested `85%/15%` three-stage hybrid `P5_Integrated_A_C_B` (`5.965%` Median RPD, `14/0/2` wins for `P3` vs `P5`, $p_{\text{Holm}} = 1.709 \times 10^{-3}$).

Note that `C02` is the Phase C winner among memetic GA configurations (`C01`–`C11`), whereas **`P3_FrozenA_FrozenB` (`A11` + `B21`)** is the best-performing integrated pipeline overall. Similarly, `P5_Integrated_A_C_B` reflects the specifically evaluated `85%/15%` allocation (`C02` with `steepest_descent + N5` at $b_{\text{LS}}=50$, followed by `B21` SA polish) rather than claiming every possible three-component budget split is suboptimal.

---

## 2. Experimental Protocol, Scale & Hardware Environment

All experiments target deterministic classical rectangular JSSP ($\min C_{\max}$) using the canonical Operation-Based Representation (OBR), deterministic active Serial SGS decoder (`D01-A`), and exact $O(L)$ disjunctive DAG evaluation (`D03-C`). Every phase strictly separated **Development (`16` instances)** for configuration selection from **Held-Out (`16` instances)** for confirmatory evaluation, with `20` frozen SHA-256 master seeds (`jssp-research-v1-sha256-20`, `ChaCha20Rng`).

| Benchmark / Execution Scope | Total Runs |
|---|---:|
| Phase A Layer A Standalone Initialization (`242` verified Track 1 instances) | `74,778` |
| Phase A Tier 2 Classical GA (`C01`, `32` Common Suite instances × `24` configs × `20` seeds) | `15,360` |
| Phase B Metaheuristics (`32` instances × `28` active configs × `20` seeds, budget $100L$) | `17,920` |
| Phase C Memetic & Dual-View GA (`32` instances × `11` configs × `20` seeds, budget $B_{\text{GA}}$) | `7,040` |
| Final Integrated Pipeline (`P1`–`P5`, `32` instances × `5` pipelines × `20` seeds) | `3,200` |
| **Total Recorded Runs** | **`118,298`** |

**Verified Execution Environment** ([final_pipeline_manifest.json](file:///home/abdelhalimyasser/Downloads/jssp-research/results/final_pipeline/manifests/final_pipeline_manifest.json)):
- **CPU:** `11th Gen Intel(R) Core(TM) i5-1145G7 @ 2.60GHz` (`4` physical / `8` logical cores)
- **Memory & OS:** `6872371200` bytes (~`6.4 GiB`) RAM, `Linux 7.0.0-43-generic x86_64`
- **Toolchain:** Python `3.15.2`, `uv 0.13.9`, `rustc 1.100.0`, `cargo 1.100.0`, `PyO3 0.30.1`, `maturin 1.16.0` (release native extension `jssp-core/0.1.0`)

---

## 3. Sequential Phase Findings

### 3.1 Phase A — Initial Population Generation (`A01`–`A24`)
- **Winner Selection (Development):** **`A11` (`Scrambled Sobol + MWR (alpha=1.0)`)**, winner lock SHA-256 `90269dbd983236d8cbf0210e33fa470c6e1d4716597597d608a259456ab125c8` ([phase_a_summary.md](file:///home/abdelhalimyasser/Downloads/jssp-research/reports/phase_a_summary.md)).
- **Development (`16` instances):** `init_best_rpd_mean = 19.464%`, `init_obr_hamming_div_mean = 0.9252`, `ga_median_rpd_mean = 7.422%`, `ga_best_rpd_mean = 5.139%`, `friedman_ga_rank = 7.28 / 24`, `893.6 ms`.
- **Held-Out (`16` instances):** `ga_median_rpd_mean = 8.252%`, `ga_best_rpd_mean = 5.325%`, `init_obr_hamming_div_mean = 0.9306`, `friedman_ga_rank = 6.84 / 24`, `832.1 ms`.
- **Statistical Significance (Blocked Friedman on GA Median RPD):** Development $\chi^2_F = 153.25$ ($p = 3.132 \times 10^{-21}$, Kendall's $W = 0.4164$); Held-Out $\chi^2_F = 199.60$ ($p = 4.176 \times 10^{-30}$).
- **QMC Scrambling Sensitivity:** Unscrambled Halton configurations (`A04`, `A08`, `A17`, `A18`) suffered initial diversity collapse (`0.561`–`0.626`) and poor downstream GA performance (Dev GA Median RPD `16.47%`–`16.87%`, Held-Out `19.54%`–`20.06%`), whereas Halton scrambling (`A05`: `7.982%` Dev / `9.325%` Held-Out) and Scrambled Sobol + MWR achieved high diversity and top downstream GA results.

### 3.2 Phase B — Local Search & Metaheuristics (`B01`–`B24, B29`–`B32`)
- **Scope:** `28` active configurations evaluated from matched `A11` starting schedules under a shared $100L$ candidate evaluation budget; `B25`–`B28` (standalone Metropolis–Hastings) were removed prior to execution per `D04-A`. `rejected_cyclic_moves` remained `0.0` across all configurations ([phase_b_summary.md](file:///home/abdelhalimyasser/Downloads/jssp-research/reports/phase_b_summary.md)).
- **Winner Selection (Development):** **`B21` (`Simulated Annealing (SA) + N5`)**, winner lock SHA-256 `3951b6d9f41c2396b3fab2420e258d6f20b8677a6ae7402f41c85149b0350650`.
  - **Development (`16`):** `median_rpd_mean = 3.951%`, `best_rpd_mean = 1.975%`, `friedman_rank = 4.34 / 28`, `mean_elapsed_ms = 261.0`.
  - **Held-Out (`16`):** `median_rpd_mean = 4.349%`, `best_rpd_mean = 2.025%`, `improvement_pct_mean = 14.80%`, `friedman_rank = 2.88 / 28`, `mean_elapsed_ms = 258.8`.
- **Top Development Runners-Up (Median RPD):** `B22 (SA + N6)` `4.370%`, `B13 (Tabu + N5)` `4.471%`, `B23 (SA + N7)` `5.064%`, `B14 (Tabu + N6)` `5.502%`, `B30 (ILS-MH + N6)` `6.298%`, `B17 (ILS + N5)` `6.358%`.
- **Neighborhood Analysis:** Within standalone descents (`B01`–`B12`), `N8` led (`B12`/`B04`: `15.040%` Dev Med RPD vs `B01`: `16.013%`), while marginal Development Best RPD across all metaheuristics ranked `N6` (`9.738%`), `N5` (`9.913%`), `N7` (`10.173%`), `N8` (`11.194%`), with `SA + N5` (`B21`) winning overall.
- **Statistical Significance (Blocked Friedman):** Development $\chi^2_F = 342.60$ ($p = 2.119 \times 10^{-56}$, $W = 0.7931$); Held-Out $\chi^2_F = 396.47$ ($p = 2.610 \times 10^{-67}$).

### 3.3 Phase C — Memetic & Dual-View Genetic Algorithms (`C01`–`C11`)
- **Embedded Local Search Specification (`docs/experiments/phase_c.md` §3):** All Phase C variants initialized from `A11` under an equalized total evaluation budget $B_{\text{GA}} = \mathrm{clamp}(64\lceil 2.5JM\rceil, 6400, 64000)$ (`64` genotypes across every configuration). Embedded local search in `C02`–`C11` used **`steepest_descent` + `N5`** capped at **$b_{\text{LS}} = 50$ evaluations per offspring decode**, deliberately distinct from full `B21` (`SA + N5`, which calibrates $T_0$ over 50 probes and would exhaust the entire `b_LS = 50` cap per individual).
- **Numerical Winner ($C^*$):** **`C02` (`Lamarckian GA (LGA)`)**, winner lock SHA-256 `f6764fb58b76e6755ba2868a1974e1aec7a4dd8a9b46061a8bd18d6c616e90bc` ([phase_c_summary.md](file:///home/abdelhalimyasser/Downloads/jssp-research/reports/phase_c_summary.md)).

| Config | Variant | Dev Med RPD | Dev Best RPD | Dev Rank | HO Med RPD | HO Best RPD | HO Rank | HO Gate $p_T$ | HO ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`C02`** | **Lamarckian GA (LGA)** | **`7.570%`** | `5.701%` | **`2.56`** | **`7.845%`** | `5.463%` | **`1.25`** | `—` | **`448.5`** |
| `C01` | Classical GA Baseline | `7.823%` | **`5.002%`** | `5.62` | `8.548%` | **`5.174%`** | `5.72` | `—` | `810.5` |
| `C04` | Partial Lamarckian GA (50%) | `8.299%` | `6.088%` | `3.81` | `8.849%` | `6.370%` | `3.47` | `—` | `570.0` |
| `C10` | Unpaired Dual-Pop Control | `8.353%` | `6.173%` | `4.19` | `8.844%` | `6.221%` | `3.59` | `—` | `541.0` |
| `C08` | Dual-View Fixed Gate (`p=0.75`) | `8.555%` | `6.133%` | `4.88` | `9.221%` | `6.283%` | `5.09` | `0.750` | `575.5` |
| `C09` | Dual-View Adaptive (`w_div=0.00`) | `8.735%` | `6.355%` | `6.25` | `9.601%` | `6.812%` | `6.69` | `0.523` | `556.6` |
| `C06` | Dual-View Adaptive (`w_div=0.30`) | `8.943%` | `6.674%` | `6.75` | `9.472%` | `6.695%` | `6.09` | `0.517` | `524.3` |
| `C11` | Dual-View + Migration (Expl.) | `8.973%` | `6.900%` | `6.91` | `9.595%` | `6.935%` | `6.97` | `0.514` | `559.3` |
| `C05` | Dual-View Fixed Gate (`p=0.50`) | `9.027%` | `6.589%` | `7.09` | `9.740%` | `6.611%` | `7.12` | `0.500` | `561.4` |
| `C07` | Dual-View Fixed Gate (`p=0.25`) | `9.584%` | `7.317%` | `8.66` | `10.581%` | `7.593%` | `9.12` | `0.250` | `538.0` |
| `C03` | Baldwinian GA (BGA) | `10.318%` | `8.409%` | `9.28` | `11.701%` | `8.854%` | `10.88` | `—` | `540.4` |

- **Statistical Significance (Blocked Friedman):** Dev $\chi^2_F = 81.82$ ($p = 2.204 \times 10^{-13}$, $W = 0.5114$); Held-Out $\chi^2_F = 105.60$ ($p = 4.106 \times 10^{-18}$, $W = 0.6600$).
- **Key Holm–Wilcoxon Findings ([phase_c_statistical_tests.csv](file:///home/abdelhalimyasser/Downloads/jssp-research/results/phase_c/analysis/phase_c_statistical_tests.csv)):**
  - `C02` is the numerical winner, but **`C02` vs `C01` was not statistically significant after Holm correction** (`9/4/3`, $p_{\text{Holm}} = 0.479$ Dev; `12/0/4`, $p_{\text{Holm}} = 0.477$ Held-Out), while `C01` attained lower mean Best RPD (`5.002%` Dev / `5.174%` Held-Out).
  - Pure Baldwinian learning (`C03`) significantly underperformed `C01` (`1/0/15` Held-Out, $p_{\text{Holm}} = 7.629 \times 10^{-4}$).
  - `C02` significantly outperformed both Fixed (`C05`) and Adaptive (`C06`) Dual-View variants on every Held-Out instance (`0/0/16`, $p_{\text{Holm}} = 1.831 \times 10^{-4}$).
  - Within Dual-View variants on Held-Out, adaptive gating (`C06`) significantly outperformed `C03` (`16/0/0`, $p_{\text{Holm}} = 1.831 \times 10^{-4}$) and `C07` (`13/2/1`, $p_{\text{Holm}} = 5.662 \times 10^{-3}$), whereas the diversity signal ablation (`C06` vs `C09`, $p_{\text{Holm}} = 0.363$) and migration (`C11` vs `C06`, $p_{\text{Holm}} = 0.300$) were not significant.
  - **Dual-View Diagnostic Synthesis (full 10-factor audit in [phase_c_summary.md](file:///home/abdelhalimyasser/Downloads/jssp-research/reports/phase_c_summary.md)):** Equalizing $B_{\text{GA}}$ under `b_LS = 50` limited memetic runs to `11.85`–`13.23` Held-Out generations (`96.9%`–`97.5%` LS share), spent ~50% of evaluations on the weaker Baldwinian view ($G^L$ beat $G^B$ in `294/320` `C05` and `290/320` `C06` Held-Out runs by `4.95%`–`5.34%`), diluted tournament pressure via Model A's pair score $p_t S_i^L + (1-p_t) S_i^B$ (while unpaired `C10` significantly beat `C05`/`C06`), left `C06`'s EWMA gate near `p_0 = 0.50` (`0.5172 ± 0.0185` Held-Out; `C09` `0.5235 ± 0.0251`), and occurred in a regime where `C02` retained high OBR Hamming diversity (`0.826` Dev / `0.831` Held-Out) without premature collapse.

---

## 4. Final Integrated Pipeline Comparison (`P1`–`P5`)

Source data: [final_dev_summary.csv](file:///home/abdelhalimyasser/Downloads/jssp-research/results/final_pipeline/analysis/final_dev_summary.csv), [final_held_out_summary.csv](file:///home/abdelhalimyasser/Downloads/jssp-research/results/final_pipeline/analysis/final_held_out_summary.csv), [final_statistical_tests.json](file:///home/abdelhalimyasser/Downloads/jssp-research/results/final_pipeline/analysis/final_statistical_tests.json), [final_family_breakdown.csv](file:///home/abdelhalimyasser/Downloads/jssp-research/results/final_pipeline/analysis/final_family_breakdown.csv).

| Pipeline | Composition | Dev Med RPD | Dev Best RPD | Dev Hit BKS | Dev Rank | Dev ms | HO Med RPD | HO Best RPD | HO Hit BKS | HO Rank | HO ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `P1_Baseline_A01_C01` | Random (`A01`) + Classical GA (`C01`) | `7.920%` | `5.116%` | `0.234` | `4.09` | `848.1` | `8.959%` | `5.641%` | `0.000` | `4.56` | `794.3` |
| `P2_FrozenA_C01` | Frozen $A^*$ (`A11`) + Classical GA (`C01`) | `7.885%` | `4.806%` | `0.250` | `3.72` | `847.7` | `8.380%` | `5.460%` | `0.000` | `3.94` | `787.3` |
| **`P3_FrozenA_FrozenB`** | **Frozen $A^*$ (`A11`) + Frozen $B^*$ (`B21` SA+`N5`)** | **`3.782%`** | **`2.156%`** | `0.241` | **`1.56`** | `544.4` | **`3.959%`** | **`1.742%`** | **`0.022`** | **`1.25`** | `532.0` |
| `P4_FrozenA_FrozenC` | Frozen $A^*$ (`A11`) + Frozen $C^*$ (`C02` LGA) | `7.888%` | `5.953%` | `0.250` | `3.38` | `509.3` | `7.960%` | `5.926%` | `0.016` | `3.09` | `507.4` |
| `P5_Integrated_A_C_B` | `A11` + `C02` (`85%`) + `B21` Polish (`15%`) | `5.834%` | `3.996%` | `0.250` | `2.25` | `549.4` | `5.965%` | `4.233%` | `0.003` | `2.16` | `576.1` |

### 4.1 Friedman & Holm–Wilcoxon Statistical Tests (`P1`–`P5`)

- **Blocked Friedman:** Development $\chi^2_F = 38.41$ ($p = 9.222 \times 10^{-8}$, Kendall's $W = 0.6002$); Held-Out $\chi^2_F = 45.61$ ($p = 2.976 \times 10^{-9}$, Kendall's $W = 0.7126$).
- **Held-Out Holm–Wilcoxon Contrasts (`16` Held-Out instances):**
  - `P2_FrozenA_C01` vs `P1_Baseline_A01_C01`: `12/0/4` ($p_{\text{raw}} = 0.0335$, $p_{\text{Holm}} = 0.0671$, $A_{12} = 0.555$)
  - `P3_FrozenA_FrozenB` vs `P2_FrozenA_C01`: **`16/0/0`** ($p_{\text{Holm}} = 2.136 \times 10^{-4}$, $A_{12} = 0.863$, Cliff's $d = 0.727$)
  - `P4_FrozenA_FrozenC` vs `P2_FrozenA_C01`: `11/0/5` ($p_{\text{raw}} = 0.348$, $p_{\text{Holm}} = 0.348$, $A_{12} = 0.551$)
  - `P5_Integrated_A_C_B` vs `P1_Baseline_A01_C01`: **`16/0/0`** ($p_{\text{Holm}} = 2.136 \times 10^{-4}$, $A_{12} = 0.684$)
  - `P5_Integrated_A_C_B` vs `P2_FrozenA_C01`: **`16/0/0`** ($p_{\text{Holm}} = 2.136 \times 10^{-4}$, $A_{12} = 0.664$)
  - `P5_Integrated_A_C_B` vs `P4_FrozenA_FrozenC`: **`11/1/4`** ($p_{\text{Holm}} = 0.0374$, $A_{12} = 0.592$)
  - `P5_Integrated_A_C_B` vs `P3_FrozenA_FrozenB`: **`2/0/14`** ($p_{\text{Holm}} = 1.709 \times 10^{-3}$, $A_{12} = 0.328$) — full-budget `P3` (`A11 + B21`) significantly outperforms the tested `85%/15%` split between Stage-1 LGA and Stage-2 SA polishing (`P5`).

### 4.2 Per-Family Mean Best RPD for `P3_FrozenA_FrozenB` ([final_family_breakdown.csv](file:///home/abdelhalimyasser/Downloads/jssp-research/results/final_pipeline/analysis/final_family_breakdown.csv))

| Family (`n_runs`) | `P1` Best RPD | `P2` Best RPD | **`P3` Best RPD** | `P4` Best RPD | `P5` Best RPD |
|---|---:|---:|---:|---:|---:|
| `ABZ` (`60`) | `6.44%` | `5.84%` | **`2.39%`** | `4.07%` | `3.32%` |
| `DMU` (`60`) | `13.63%` | `12.87%` | **`5.59%`** | `13.20%` | `10.07%` |
| `FT` (`60`) | `4.11%` | `3.99%` | **`2.47%`** | `4.62%` | `2.84%` |
| `LA` (`180`) | `4.46%` | `4.15%` | **`1.92%`** | `3.12%` | `2.90%` |
| `ORB` (`80`) | `7.19%` | `6.81%` | **`3.30%`** | `4.92%` | `4.65%` |
| `SWV` (`80`) | `14.98%` | `15.48%` | **`9.59%`** | `20.00%` | `12.15%` |
| `TA` (`80`) | `11.69%` | `10.77%` | **`4.44%`** | `9.63%` | `7.47%` |
| `YN` (`40`) | `13.57%` | `12.17%` | **`4.30%`** | `10.19%` | `9.60%` |

---

## 5. Experimental Provenance & Reproducibility Audit

1. **Statistical Regeneration & Winner-Lock Digest Verification:**
   - All four frozen winner files (`frozen_winner_a.json`, `frozen_winner_b.json`, `frozen_winner_c.json`, `frozen_final_pipeline.json`) match their recorded `winner_lock_sha256` under `canonical_json_sha256`.
   - All blocked Friedman test statistics across Development (`16`) and Held-Out (`16`) for Phase A, Phase B, Phase C, and Final Pipeline (`P1`–`P5`) regenerate identically from the stored raw JSONL files (`layer_a_raw_sha256` = `7cf110f0...`, `tier2_raw_sha256` = `c20bb3f1...`, Phase B `raw_jsonl_sha256` = `0b752b41...`, Phase C `raw_jsonl_sha256` = `98dcfd85...`, Final `raw_jsonl_sha256` = `db35c16f...`).
2. **Audit of `git.dirty = true` in Historical Phase Manifests (Preserved Unmodified):**
   - **Phase B (`commit = 88ed4571...`, `diff_sha256 = b33f6a21...`):** Forensically verified exact match against `git status --porcelain=v1` showing **zero** modified tracked files (`diff HEAD` empty); `dirty = true` occurred solely because the runner created untracked `results/phase_b/` before capturing Git provenance (`commit` `4c327e6e` immediately committed those outputs cleanly).
   - **Phase C (`commit = 4c327e6e...`, `diff_sha256 = 74472d86...`):** Forensically verified **exact cryptographic SHA-256 match** against the diff between `4c327e6e` and `18117347` (`crates/jssp-python/src/phase_c_api.rs`, `scripts/run_phase_c_pipeline.py`, `scripts/run_final_pipeline.py` plus untracked `results/phase_c/`). Source tree matches `18117347` verbatim.
   - **Final Pipeline (`commit = 18117347...`, `diff_sha256 = a949750c...`):** Forensically verified **exact cryptographic SHA-256 match** against the diff between `18117347` and `f22ece5a` (`scripts/run_final_pipeline.py` plus untracked `results/final_pipeline/`). Source tree matches `f22ece5a` verbatim.
   - **Phase A (`commit = 12d0de87...`, `diff_sha256 = 14765aeb...`):** Executed with Phase B/C development files and untracked artifacts in the working tree prior to `88ed4571`. Because intermediate edits occurred between Phase A execution and commit `88ed4571`, the exact working-tree snapshot `14765aeb...` cannot be byte-for-byte reconstructed solely from Git history—explicitly disclosed as a provenance limitation on Phase A's execution commit, alongside its intact raw JSONL SHA-256 digests (`7cf110f0...`, `c20bb3f1...`).
