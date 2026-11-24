# Phase B Summary — Local Search & Metaheuristics (`B01`–`B24`, `B29`–`B32`)

- **Frozen Starting Initializer ($A^*$):** `A11` (matched starting schedules per instance $\times$ frozen seed).
- **Selected & Frozen Winner ($B^*$):** **`B21` — Simulated Annealing (SA) + N5**
- **Development Split Winner Selection (16 instances):** Mean instance-level median `RPD_BKS` = `3.951%`, Mean Best `RPD_BKS` = `1.975%`, Friedman rank = `4.34` / 28.
- **Held-Out Generalization ($B^*$, 16 held-out instances):** Mean instance-level median `RPD_BKS` = `4.349%`, Mean Best `RPD_BKS` = `2.025%`, Mean Improvement = `14.80%`.
- **Experimental Volume & Protocol:** `17,920` runs (`32` Common Suite instances $\times$ `28` active configs $\times$ `20` frozen seeds; budget `100 * L`; `B25`–`B28` excluded per `D04-A`).
- **Blocked Friedman Test:** Dev $\chi^2_F = 342.60$ ($p = 2.119e-56$, Kendall $W = 0.7931$); Held-Out $\chi^2_F = 396.47$ ($p = 2.610e-67$).

## All 28 Active Phase B Configurations — Development & Held-Out Summary

| Rank (Dev) | ID | Strategy | Nbh | Dev Med RPD | Dev Best RPD | Dev Rank | Held-Out Med RPD | Held-Out Best RPD | Anytime RPD AUC (Dev) | Mean ms |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `B21` | Simulated Annealing (SA) | `N5` | 3.951% | 1.975% | 4.34 | 4.349% | 2.025% | 6.225% | 261.0 |
| 2 | `B22` | Simulated Annealing (SA) | `N6` | 4.370% | 2.337% | 5.06 | 4.524% | 2.410% | 8.397% | 360.0 |
| 3 | `B13` | Tabu Search | `N5` | 4.471% | 2.778% | 4.75 | 4.665% | 2.510% | 6.014% | 181.8 |
| 4 | `B23` | Simulated Annealing (SA) | `N7` | 5.064% | 2.606% | 7.31 | 5.192% | 2.987% | 9.955% | 337.5 |
| 5 | `B14` | Tabu Search | `N6` | 5.502% | 2.974% | 7.56 | 5.739% | 2.784% | 7.260% | 161.9 |
| 6 | `B30` | Metropolis-Filtered ILS + Steepest Descent | `N6` | 6.298% | 4.067% | 8.19 | 6.769% | 4.404% | 7.497% | 224.8 |
| 7 | `B17` | Iterated Local Search (ILS) | `N5` | 6.358% | 4.156% | 9.16 | 6.648% | 3.808% | 7.638% | 183.7 |
| 8 | `B18` | Iterated Local Search (ILS) | `N6` | 6.405% | 4.006% | 9.16 | 6.519% | 3.707% | 7.822% | 175.7 |
| 9 | `B19` | Iterated Local Search (ILS) | `N7` | 6.460% | 4.249% | 8.97 | 6.881% | 3.839% | 7.886% | 166.6 |
| 10 | `B31` | Metropolis-Filtered ILS + Steepest Descent | `N7` | 6.499% | 4.626% | 8.88 | 7.164% | 4.615% | 7.666% | 194.4 |
| 11 | `B29` | Metropolis-Filtered ILS + Steepest Descent | `N5` | 6.753% | 4.814% | 9.81 | 7.393% | 4.798% | 7.770% | 211.0 |
| 12 | `B32` | Metropolis-Filtered ILS + Steepest Descent | `N8` | 7.257% | 5.043% | 11.19 | 8.116% | 4.990% | 8.814% | 128.6 |
| 13 | `B15` | Tabu Search | `N7` | 7.562% | 4.410% | 12.12 | 8.365% | 4.241% | 9.186% | 146.3 |
| 14 | `B20` | Iterated Local Search (ILS) | `N8` | 7.705% | 4.814% | 12.38 | 8.330% | 5.168% | 9.323% | 139.1 |
| 15 | `B24` | Simulated Annealing (SA) | `N8` | 8.378% | 5.370% | 13.50 | 10.023% | 6.129% | 15.384% | 364.8 |
| 16 | `B16` | Tabu Search | `N8` | 9.591% | 6.225% | 14.62 | 11.171% | 6.054% | 10.861% | 127.1 |
| 17 | `B12` | Random Improvement | `N8` | 15.040% | 10.289% | 20.19 | 17.670% | 11.884% | 15.047% | 1.3 |
| 18 | `B04` | Steepest Descent | `N8` | 15.040% | 10.216% | 19.06 | 18.231% | 12.408% | 15.087% | 3.8 |
| 19 | `B10` | Random Improvement | `N6` | 15.068% | 10.224% | 19.94 | 18.179% | 12.663% | 15.071% | 0.7 |
| 20 | `B08` | First Improvement | `N8` | 15.119% | 10.053% | 19.59 | 18.048% | 12.385% | 15.136% | 2.3 |
| 21 | `B11` | Random Improvement | `N7` | 15.172% | 10.338% | 20.34 | 18.198% | 12.071% | 15.176% | 0.8 |
| 22 | `B02` | Steepest Descent | `N6` | 15.338% | 10.123% | 21.12 | 18.562% | 12.688% | 15.347% | 0.9 |
| 23 | `B03` | Steepest Descent | `N7` | 15.338% | 10.123% | 21.12 | 18.562% | 12.688% | 15.351% | 1.3 |
| 24 | `B06` | First Improvement | `N6` | 15.424% | 10.373% | 21.31 | 18.420% | 12.488% | 15.429% | 1.1 |
| 25 | `B07` | First Improvement | `N7` | 15.424% | 10.373% | 21.31 | 18.420% | 12.488% | 15.430% | 1.2 |
| 26 | `B09` | Random Improvement | `N5` | 15.898% | 11.096% | 25.31 | 19.219% | 13.994% | 15.900% | 0.5 |
| 27 | `B05` | First Improvement | `N5` | 15.923% | 10.851% | 24.09 | 19.098% | 13.770% | 15.925% | 0.7 |
| 28 | `B01` | Steepest Descent | `N5` | 16.013% | 10.837% | 25.59 | 19.103% | 13.990% | 16.018% | 0.6 |

## Neighborhood Marginal Comparison (`N5`–`N8`)

| Neighborhood | Dev Best RPD | Held-Out Best RPD | Mean Improvement | Cycle Rejects/Run | Mean ms |
|---|---:|---:|---:|---:|---:|
| `N6` | 9.738% | 11.249% | 8.49% | 0.0 | 137.9 |
| `N5` | 9.913% | 11.516% | 8.31% | 0.0 | 120.0 |
| `N7` | 10.173% | 11.837% | 8.08% | 0.0 | 120.5 |
| `N8` | 11.194% | 13.111% | 7.16% | 0.0 | 110.6 |

## Key Artifacts
- Frozen Winner $B^*$: [`results/phase_b/winner/frozen_winner_b.json`](file:///home/abdelhalimyasser/Downloads/jssp-research/results/phase_b/winner/frozen_winner_b.json)
- Manifest: [`results/phase_b/manifests/phase_b_manifest.json`](file:///home/abdelhalimyasser/Downloads/jssp-research/results/phase_b/manifests/phase_b_manifest.json)
- Statistical Tests: [`results/phase_b/analysis/phase_b_statistical_tests.json`](file:///home/abdelhalimyasser/Downloads/jssp-research/results/phase_b/analysis/phase_b_statistical_tests.json)
