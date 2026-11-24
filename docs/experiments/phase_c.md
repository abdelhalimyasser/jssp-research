# Phase C Experiment Protocol — Classical, Memetic & Dual-View Genetic Algorithms (`D06`, `D07`)

**Status:** Frozen Specification (`C01`–`C11`, **11 Configurations**)
**Algorithmic Specifications:** [`docs/algorithms/dual_view.md`](file:///home/abdelhalimyasser/Downloads/jssp-research/docs/algorithms/dual_view.md), [`docs/algorithms/obr_decoder.md`](file:///home/abdelhalimyasser/Downloads/jssp-research/docs/algorithms/obr_decoder.md)

---

## 1. Frozen Classical GA Baseline (`C01`) Parameters (`D07-A`)

All Phase C variants (`C01`–`C11`) and Phase A Stage 2 share identical baseline GA operators:
- **Representation & Decoder:** OBR (`L = \sum m_j`) decoded by Serial SGS with earliest feasible idle-gap insertion (`D01-A`).
- **Genotype Capacity:** Total **$N = 64$ OBR genotypes** ($N = 64$ individuals for `C01`–`C04`; $N_{\text{pairs}} = 32$ paired individuals $= 64$ genotypes for `C05`–`C09`, `C11`; $32 + 32 = 64$ genotypes for `C10`).
- **Initialization:** Uniform Random (`A01`) unless testing Phase A's screening winner; matched identical initial populations per `(instance, seed)`.
- **Parent Selection:** Deterministic binary tournament ($k = 2$, `selection` stream; tie-break by smaller individual/pair ID).
- **Crossover:** **Precedence Operation Crossover (POX)** / Job-based Order Crossover preserving OBR job multiplicities ($p_c = 0.85$, `crossover` stream):
  1. Randomly partition job set $J$ into non-empty subset $J_1$ and complement $J \setminus J_1$ (each job assigned to $J_1$ with probability $0.5$, rejecting trivial all/none split).
  2. Copy all genes belonging to $J_1$ from Parent 1 at their exact indices; fill remaining empty positions left-to-right with genes belonging to $J \setminus J_1$ in their exact relative order from Parent 2 (swap Parent 1/2 to form second child).
- **Mutation:** With probability $p_m = 0.15$ (`mutation` stream), draw $50\%$ chance of **OBR Swap** (two positions with distinct job IDs) or $50\%$ chance of **OBR Insertion** (move gene at index $i$ to index $j$ with distinct job IDs).
- **Elitism & Replacement:** Generational replacement preserving top $E = 2$ genotypes ($E_{\text{pairs}} = 1$ pair in Dual-View).

---

## 2. Phase C Configuration Matrix (`C01`–`C11`, 11 Configurations)

| ID | Method | Genotype Capacity | Learning / Gate Rule | Local Search Cap |
|---|---|---:|---|---|
| `C01` | Classical GA | 64 | None (non-memetic reference) | 0 |
| `C02` | Lamarckian GA (LGA) | 64 | $100\%$ genotype write-back after LS | $b_{\text{LS}} = 50$ evals/ind |
| `C03` | Baldwinian GA (BGA) | 64 | $0\%$ write-back; fitness from LS phenotype | $b_{\text{LS}} = 50$ evals/ind |
| `C04` | Partial Lamarckian GA (PLGA) | 64 | $50\%$ write-back: sort population by post-LS $C_{\max}$ (tie: ID); write back **even ranks** ($0, 2, \dots, 62$), keep **odd ranks** Baldwinian | $b_{\text{LS}} = 50$ evals/ind |
| `C05` | Dual-View Fixed Gate ($p = 0.50$) | 32 pairs (64) | Model A fixed gate $p_t \equiv 0.50$ ($G^L$ weight) | $b_{\text{LS}} = 50$ evals/view |
| `C06` | Dual-View Adaptive Gate | 32 pairs (64) | Model A EWMA rule-based gate ($\Delta_{\text{perf}}, \Delta_{\text{div}}$, $w_{\text{perf}}=0.70, w_{\text{div}}=0.30$) | $b_{\text{LS}} = 50$ evals/view |
| `C07` | Dual-View Fixed Gate ($p = 0.25$) | 32 pairs (64) | Model A fixed gate $p_t \equiv 0.25$ (Baldwinian-favored) | $b_{\text{LS}} = 50$ evals/view |
| `C08` | Dual-View Fixed Gate ($p = 0.75$) | 32 pairs (64) | Model A fixed gate $p_t \equiv 0.75$ (Lamarckian-favored) | $b_{\text{LS}} = 50$ evals/view |
| `C09` | Dual-View Adaptive Gate (no div) | 32 pairs (64) | Model A adaptive ablation with $w_{\text{div}} = 0.00$ | $b_{\text{LS}} = 50$ evals/view |
| `C10` | Unpaired Dual-Population Control | 32 LGA + 32 BGA | Two independent pools; no shared $\mathrm{ID}_i$ or gate | $b_{\text{LS}} = 50$ evals/ind |
| `C11` | Cross-View Migration (Exploratory) | 32 pairs (64) | `C06` + top-3 ($10\%$) migration every 10 gens | $b_{\text{LS}} = 50$ evals/view |

---

## 3. Local Search Integration & Budget Parity
- **Embedded Local Search Operator:** Bounded **First Improvement or Steepest Descent** using the top-performing neighborhood $N^*$ selected from Phase B screening (provisionally `N7` or `N8` from Phase B screening, frozen identically across `C02`–`C11` prior to Phase C confirmatory execution).
- **Per-Individual LS Cap:** At most $b_{\text{LS}} = 50$ neighbor evaluations per genotype decode (evaluated only on newly generated non-elite offspring so elite genotypes are not repeatedly re-searched without change).
- **Equalized Total Evaluation Budget:** Every configuration (`C01`–`C11`) runs under the **identical total objective evaluation budget** $B_{\text{eval}}^{\text{C}} = 1000 \cdot L$ (confirmatory) / $200 \cdot L$ (screening). Note that under fixed $B_{\text{eval}}^{\text{C}}$, `C01` executes more generations (zero LS overhead) while `C02`–`C11` trade generations for local-search intensification—providing a completely fair equal-effort comparison.

---

## 4. Targeted Research Contrasts
1. **`C02`, `C03`, `C04` vs. `C01`:** Net benefit of memetic learning and Partial Lamarckianism under equalized evaluation cost.
2. **`C05` vs. `C02`–`C04`:** Value of persistent paired Dual-View representation at equal $50:50$ weight without adaptive feedback.
3. **`C06` vs. `C05`, `C07`, `C08`:** Value of rule-based dynamic gate adaptation vs. static weights.
4. **`C06` vs. `C09`:** Causal contribution of the diversity signal $\Delta_{\text{div}}(t)$.
5. **`C05`/`C06` vs. `C10`:** Value of shared pair identity $\mathrm{ID}_i$ and pair-level selection vs. merely splitting a population into independent LGA and BGA pools.
6. **`C11` vs. `C06`:** Exploratory impact of controlled cross-view migration.
