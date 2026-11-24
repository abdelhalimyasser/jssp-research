# Phase C Summary — Memetic & Dual-View Genetic Algorithms (`C01`–`C11`)

- **Transferred Frozen Components:** $A^* =$ `A11`, $B^* =$ `B21` (`Simulated Annealing (SA) + N5`); Embedded LS (`phase_c.md` §3): `steepest_descent + N5` ($b_{\text{LS}} = 50$)
- **Selected & Frozen Primary Winner ($C^*$):** **`C02` — Lamarckian GA (LGA)**
- **Development Split (16 instances):** Mean instance-level median `RPD_BKS` = `7.570%`, Mean Best `RPD_BKS` = `5.701%`, Friedman rank = `2.56` / 11.
- **Held-Out Generalization (16 held-out instances):** Median `RPD_BKS` = `7.845%`, Best `RPD_BKS` = `5.463%`, Hit BKS = `0.006`.
- **Experimental Volume & Friedman:** `7,040` runs (`32` instances $\times$ `11` configs $\times$ `20` seeds); Dev $\chi^2_F = 81.82$ ($p = 2.204e-13$), Held-Out $\chi^2_F = 105.60$ ($p = 4.106e-18$).

## All 11 Phase C Configurations — Development & Held-Out Summary

| Rank (Dev) | ID | Method | Dev Med RPD | Dev Best RPD | Dev Rank | Held-Out Med RPD | Held-Out Best RPD | Mean Gate p | Final Div | Mean ms |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `C02` | Lamarckian GA (LGA) | 7.570% | 5.701% | 2.56 | 7.845% | 5.463% | — | 0.826 | 454.3 |
| 2 | `C01` | Classical GA Baseline | 7.823% | 5.002% | 5.62 | 8.548% | 5.174% | — | 0.580 | 833.6 |
| 3 | `C04` | Partial Lamarckian GA (PLGA 50%) | 8.299% | 6.088% | 3.81 | 8.849% | 6.370% | — | 0.845 | 609.3 |
| 4 | `C10` | Unpaired Dual-Population Control (32 LGA + 32 BGA) | 8.353% | 6.173% | 4.19 | 8.844% | 6.221% | — | 0.867 | 572.1 |
| 5 | `C08` | Dual-View Fixed Gate (p=0.75) | 8.555% | 6.133% | 4.88 | 9.221% | 6.283% | 0.750 | 0.871 | 601.8 |
| 6 | `C09` | Dual-View Adaptive Gate Ablation (w_div=0.00) | 8.735% | 6.355% | 6.25 | 9.601% | 6.812% | 0.515 | 0.874 | 556.8 |
| 7 | `C06` | Dual-View Adaptive Gate (w_perf=0.70, w_div=0.30) | 8.943% | 6.674% | 6.75 | 9.472% | 6.695% | 0.509 | 0.873 | 562.3 |
| 8 | `C11` | Dual-View Adaptive + Cross-View Migration (Exploratory) | 8.973% | 6.900% | 6.91 | 9.595% | 6.935% | 0.506 | 0.873 | 600.7 |
| 9 | `C05` | Dual-View Fixed Gate (p=0.50) | 9.027% | 6.589% | 7.09 | 9.740% | 6.611% | 0.500 | 0.875 | 551.0 |
| 10 | `C07` | Dual-View Fixed Gate (p=0.25) | 9.584% | 7.317% | 8.66 | 10.581% | 7.593% | 0.250 | 0.878 | 577.0 |
| 11 | `C03` | Baldwinian GA (BGA) | 10.318% | 8.409% | 9.28 | 11.701% | 8.854% | — | 0.873 | 554.5 |

## Key Artifacts
- Frozen Winner $C^*$: [`results/phase_c/winner/frozen_winner_c.json`](file:///home/abdelhalimyasser/Downloads/jssp-research/results/phase_c/winner/frozen_winner_c.json)
- Statistical Tests: [`results/phase_c/analysis/phase_c_statistical_tests.json`](file:///home/abdelhalimyasser/Downloads/jssp-research/results/phase_c/analysis/phase_c_statistical_tests.json)
- Manifest: [`results/phase_c/manifests/phase_c_manifest.json`](file:///home/abdelhalimyasser/Downloads/jssp-research/results/phase_c/manifests/phase_c_manifest.json)
