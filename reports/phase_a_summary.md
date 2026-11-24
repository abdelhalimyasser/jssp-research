# Phase A Summary — Initial Population Generation (`A01`–`A24`)

- **Selected & Frozen Winner ($A^*$):** **`A11` — Scrambled Sobol + MWR (alpha=1.0)**
- **Winner Selection Rule (Dev Only, 16 instances):** Lowest mean instance-level median `C01` GA `RPD_BKS` = `7.422%` (Friedman rank `7.28` / 24).
- **Held-Out Generalization ($A^*$, 16 held-out instances):** Median GA `RPD_BKS` = `8.252%`, Mean Best `RPD_BKS` = `5.325%`, Initial OBR Diversity = `0.9306`.
- **Experimental Volume:** `74,778` standalone Layer A runs across all `242` verified Track 1 instances + `15,360` full `C01` Classical GA runs across the `32`-instance Common Suite (`16` dev + `16` held-out $\times$ `24` configs $\times$ `20` frozen seeds).
- **Blocked Friedman Test (Downstream GA Median RPD):** Dev $\chi^2_F = 153.25$ ($p = 3.132e-21$, Kendall $W = 0.4164$); Held-Out $\chi^2_F = 199.60$ ($p = 4.176e-30$).

## All 24 Configurations — Development (Winner Selection) & Held-Out Results

| Rank (Dev) | ID | Method | Init Best RPD (Dev) | Init Div (Dev) | GA Med RPD (Dev) | Dev Rank | GA Med RPD (Held-Out) | GA Best RPD (Held-Out) | Mean Time (ms) |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `A11` | Scrambled Sobol + MWR (alpha=1.0) | 19.46% | 0.925 | 7.422% | 7.28 | 8.252% | 5.325% | 893.6 |
| 2 | `A22` | Sobol + MWR (alpha=2.0) | 18.25% | 0.917 | 7.534% | 8.50 | 8.436% | 5.363% | 887.7 |
| 3 | `A10` | Sobol + MWR (alpha=1.0) | 19.76% | 0.925 | 7.552% | 9.50 | 7.950% | 5.098% | 886.3 |
| 4 | `A21` | Sobol + MWR (alpha=0.5) | 20.26% | 0.930 | 7.587% | 9.12 | 8.335% | 5.282% | 842.0 |
| 5 | `A03` | Monte Carlo + MWR (alpha=1.0) | 19.71% | 0.918 | 7.672% | 10.00 | 8.392% | 5.443% | 832.1 |
| 6 | `A09` | Scrambled Halton + MWR (alpha=1.0) | 19.48% | 0.921 | 7.697% | 9.59 | 8.485% | 5.449% | 845.9 |
| 7 | `A19` | Scrambled Halton + MWR (alpha=0.5) | 20.71% | 0.926 | 7.768% | 10.59 | 8.646% | 5.599% | 894.4 |
| 8 | `A15` | Monte Carlo + MWR (alpha=0.5) | 20.53% | 0.922 | 7.781% | 9.47 | 8.713% | 5.541% | 836.1 |
| 9 | `A23` | Scrambled Sobol + MWR (alpha=0.5) | 20.55% | 0.930 | 7.848% | 10.81 | 8.636% | 5.497% | 866.7 |
| 10 | `A14` | Mixed Scrambled Halton/Sobol (75:25) | 23.20% | 0.928 | 7.910% | 10.81 | 9.080% | 5.364% | 860.3 |
| 11 | `A13` | Mixed Scrambled Halton/Sobol (25:75) | 23.30% | 0.931 | 7.958% | 11.66 | 9.041% | 5.891% | 880.7 |
| 12 | `A05` | Scrambled Halton (alpha=0.0) | 23.32% | 0.928 | 7.982% | 11.88 | 9.325% | 5.875% | 867.8 |
| 13 | `A20` | Scrambled Halton + MWR (alpha=2.0) | 18.85% | 0.914 | 8.007% | 10.97 | 8.320% | 5.224% | 849.5 |
| 14 | `A12` | Mixed Scrambled Halton/Sobol (50:50) | 23.36% | 0.930 | 8.062% | 12.19 | 9.051% | 5.871% | 808.1 |
| 15 | `A24` | Scrambled Sobol + MWR (alpha=2.0) | 18.51% | 0.917 | 8.063% | 11.12 | 8.517% | 5.439% | 831.5 |
| 16 | `A16` | Monte Carlo + MWR (alpha=2.0) | 18.69% | 0.911 | 8.083% | 11.62 | 8.616% | 5.522% | 885.5 |
| 17 | `A01` | Uniform Random (alpha=0.0) | 23.23% | 0.924 | 8.138% | 12.50 | 9.135% | 5.378% | 900.7 |
| 18 | `A07` | Scrambled Sobol (alpha=0.0) | 23.13% | 0.933 | 8.221% | 13.28 | 9.203% | 5.531% | 889.2 |
| 19 | `A06` | Unscrambled Sobol (alpha=0.0) | 24.40% | 0.932 | 8.365% | 12.00 | 9.532% | 6.257% | 863.3 |
| 20 | `A02` | Deterministic MWR | 26.78% | 0.000 | 10.709% | 16.84 | 11.813% | 7.967% | 857.8 |
| 21 | `A18` | Halton + MWR (alpha=2.0) | 29.66% | 0.626 | 16.470% | 19.38 | 19.627% | 14.620% | 815.0 |
| 22 | `A17` | Halton + MWR (alpha=0.5) | 31.34% | 0.572 | 16.549% | 19.72 | 20.063% | 14.577% | 785.2 |
| 23 | `A08` | Halton + MWR (alpha=1.0) | 29.68% | 0.561 | 16.583% | 20.06 | 19.833% | 14.640% | 789.1 |
| 24 | `A04` | Unscrambled Halton (alpha=0.0) | 31.02% | 0.605 | 16.872% | 21.09 | 19.537% | 14.567% | 771.9 |

## Key Artifacts
- Frozen Winner Lock (`90269dbd983236d8...`): [`results/phase_a/winner/frozen_winner_a.json`](file:///home/abdelhalimyasser/Downloads/jssp-research/results/phase_a/winner/frozen_winner_a.json)
- Statistical Tests (Friedman + Holm-Wilcoxon + A12 + Cliff's d): [`results/phase_a/analysis/phase_a_friedman_and_wilcoxon.json`](file:///home/abdelhalimyasser/Downloads/jssp-research/results/phase_a/analysis/phase_a_friedman_and_wilcoxon.json)
- Full 242-Instance Layer A Summary: [`results/phase_a/analysis/phase_a_layer_a_242_summary.csv`](file:///home/abdelhalimyasser/Downloads/jssp-research/results/phase_a/analysis/phase_a_layer_a_242_summary.csv)
- Manifest Index: [`results/phase_a/manifests/phase_a_manifest_index.json`](file:///home/abdelhalimyasser/Downloads/jssp-research/results/phase_a/manifests/phase_a_manifest_index.json)
