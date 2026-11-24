# Phase A Experiment Protocol — Initial Population Generation (`D02`, `D07`)

**Status:** Frozen Specification (`A01`–`A24`, **24 Configurations**)
**Algorithmic Specification:** [`docs/algorithms/qmc_mapping.md`](file:///home/abdelhalimyasser/Downloads/jssp-research/docs/algorithms/qmc_mapping.md), [`docs/algorithms/obr_decoder.md`](file:///home/abdelhalimyasser/Downloads/jssp-research/docs/algorithms/obr_decoder.md)

---

## 1. Experimental Design & Configuration Table

Phase A evaluates 24 initialization configurations under a common OBR representation, canonical Serial SGS active decoder, and population size $N = 64 = 2^6$.

| ID | Method | Generator | MWR $\alpha$ | Mix Ratio ($N_H : N_S$) | Deterministic? |
|---|---|---|---:|---:|---|
| `A01` | Uniform Random | PRNG (`init` stream) | 0.0 | — | Stochastic (20 seeds) |
| `A02` | Deterministic MWR | Greedy max $W_j(t)$ | $\infty$ (tie: min `JobId`) | — | Yes (1 run) |
| `A03` | Monte Carlo + MWR | PRNG (`init` stream) | 1.0 | — | Stochastic (20 seeds) |
| `A04` | Halton | Unscrambled Halton | 0.0 | — | Yes (1 run) |
| `A05` | Scrambled Halton | Digit-permuted Halton | 0.0 | — | Stochastic (20 seeds) |
| `A06` | Sobol | Unscrambled Joe–Kuo Sobol | 0.0 | — | Yes (1 run) |
| `A07` | Scrambled Sobol | Matoušek-scrambled Sobol | 0.0 | — | Stochastic (20 seeds) |
| `A08` | Halton + MWR | Unscrambled Halton | 1.0 | — | Yes (1 run) |
| `A09` | Scrambled Halton + MWR | Digit-permuted Halton | 1.0 | — | Stochastic (20 seeds) |
| `A10` | Sobol + MWR | Unscrambled Joe–Kuo Sobol | 1.0 | — | Yes (1 run) |
| `A11` | Scrambled Sobol + MWR | Matoušek-scrambled Sobol | 1.0 | — | Stochastic (20 seeds) |
| `A12` | Mixed Halton / Sobol (50:50) | Scrambled Halton ($i=1\dots 32$) + Scrambled Sobol ($i=0\dots 31$, $2^5$ net) | 0.0 | `32 : 32` | Stochastic (20 seeds) |
| `A13` | Mixed Halton / Sobol (25:75) | Scrambled Halton ($i=1\dots 16$) + Scrambled Sobol ($i=0\dots 47$, empirical non-$2^m$ prefix) | 0.0 | `16 : 48` | Stochastic (20 seeds) |
| `A14` | Mixed Halton / Sobol (75:25) | Scrambled Halton ($i=1\dots 48$) + Scrambled Sobol ($i=0\dots 15$, $2^4$ net) | 0.0 | `48 : 16` | Stochastic (20 seeds) |
| `A15` | Monte Carlo + MWR ($\alpha=0.5$) | PRNG (`init` stream) | 0.5 | — | Stochastic (20 seeds) |
| `A16` | Monte Carlo + MWR ($\alpha=2.0$) | PRNG (`init` stream) | 2.0 | — | Stochastic (20 seeds) |
| `A17` | Halton + MWR ($\alpha=0.5$) | Unscrambled Halton | 0.5 | — | Yes (1 run) |
| `A18` | Halton + MWR ($\alpha=2.0$) | Unscrambled Halton | 2.0 | — | Yes (1 run) |
| `A19` | Scrambled Halton + MWR ($\alpha=0.5$) | Digit-permuted Halton | 0.5 | — | Stochastic (20 seeds) |
| `A20` | Scrambled Halton + MWR ($\alpha=2.0$) | Digit-permuted Halton | 2.0 | — | Stochastic (20 seeds) |
| `A21` | Sobol + MWR ($\alpha=0.5$) | Unscrambled Joe–Kuo Sobol | 0.5 | — | Yes (1 run) |
| `A22` | Sobol + MWR ($\alpha=2.0$) | Unscrambled Joe–Kuo Sobol | 2.0 | — | Yes (1 run) |
| `A23` | Scrambled Sobol + MWR ($\alpha=0.5$) | Matoušek-scrambled Sobol | 0.5 | — | Stochastic (20 seeds) |
| `A24` | Scrambled Sobol + MWR ($\alpha=2.0$) | Matoušek-scrambled Sobol | 2.0 | — | Stochastic (20 seeds) |

---

## 2. Two-Stage Evaluation Protocol

1. **Stage A1 — Standalone Initializer Evaluation:**
   - Generate $N = 64$ valid OBR chromosomes and decode all $64$ via Serial SGS (`64` objective evaluations).
   - Measure **best / mean / median / SD initial $C_{\max}$**, **RPD against frozen BKS**, **feasible schedule rate** (expected `1.0`), **OBR normalized pairwise Hamming diversity** $D_{\text{Ham}}$, **chromosome duplicate rate**, **unique decoded schedule count** (unique machine sequences / makespans), and **wall-clock generation + decode time**.
2. **Stage A2 — Downstream Classical GA Impact:**
   - Feed each generated initial population directly into the frozen **Classical GA baseline (`C01`)** (identical selection, POX crossover $p_c = 0.85$, mutation $p_m = 0.15$, elitism $E = 2$, and identical total evaluation budget $B_{\text{eval}}^{\text{GA}}$ including the initial `64` decodes).
   - *(Note: For deterministic initializers `A02`, `A04`, `A06`, `A08`, `A10`, `A17`, `A18`, `A21`, `A22`, Stage A1 requires 1 deterministic run, whereas Stage A2 executes 20 independent GA seeds starting from that identical deterministic initial population.)*

---

## 3. Sequence Indexing, Balance Status & Budget Parity
- **Contiguous Sequence Prefixes:**
  - Full-population Halton (`A04`, `A05`, `A08`, `A09`, `A17`–`A20`): 1-based contiguous prefix $i \in \{1, \dots, 64\}$.
  - Full-population Scrambled Sobol (`A07`, `A11`, `A23`, `A24`): 0-based contiguous prefix $i \in \{0, \dots, 63\}$ ($N_S = 64 = 2^6$, exact $(t, 6, s)$ net).
  - Full-population Unscrambled Sobol (`A06`, `A10`, `A21`, `A22`): shifted 1-based contiguous prefix $i \in \{1, \dots, 64\}$ omitting the degenerate origin $i = 0$ (no 0-based dyadic net closure at $i = 64$).
  - Mixed QMC (`A12`–`A14`): Halton uses $i \in \{1,\dots,N_H\}$; Scrambled Sobol uses contiguous prefix $i \in \{0,\dots,N_S - 1\}$. Sample sizes $N_S = 32 = 2^5$ (`A12`) and $N_S = 16 = 2^4$ (`A14`) are powers of two with exact digital-net balance; **`A13` ($N_S = 48$) is NOT a power of two and is recorded as an empirical mixed-QMC configuration without standard digital-net balance guarantees**.
- **Dimension Cap:** Standard Sobol (`A06`, `A07`, `A10`–`A14`, `A21`–`A24`) applies strictly to instances with **$L \le 21,201$** (covers 100% of Track 1 and Track 3, plus Track 2a strata with $L \le 21,201$).
- **Budget Parity:** Total objective evaluation budget in Stage A2 charges initial population evaluation (`64`) plus every offspring decode, guaranteeing exact parity across `A01`–`A24`.
