# RESEARCH_PLAN_V1.1 — Reconciled Scientific Specification

**Status:** M1-B Reconciled & Frozen Scientific Specification
**Supersedes (for M2+ implementation):** [`RESEARCH_PLAN_V1.md`](file:///home/abdelhalimyasser/Downloads/jssp-research/docs/research/RESEARCH_PLAN_V1.md) *(preserved unmodified as the M0 audit baseline)*
**Traceability:** Every algorithmic and protocol update in V1.1 is governed by [`DECISION_LOG_M1B.md`](file:///home/abdelhalimyasser/Downloads/jssp-research/docs/research/DECISION_LOG_M1B.md).
**Primary Objective:** Minimize makespan ($C_{\max}$) in deterministic classical rectangular Job-Shop Scheduling Problems (JSSP).

---

## 1. Summary of Changes from V1.0 to V1.1

1. **Decoder & Adapter Semantics (`D01`):** Confirmed **Serial SGS with earliest feasible idle-gap insertion** generating **active schedules** (`D01-A`: `ACCEPTED`), paired with canonical topological OBR re-encoding ordered by `(r_o, r_o + p_o, job_id, op_index)` and active-left-shift telemetry (`D01-B`: `PROVISIONAL` — verified 100% `Exact` on active M2 schedules $\mathcal{S}_{\text{A}}$ and strictly non-worsening left-shift projection $S'_o \le r_o$ on non-active orientations $\mathcal{S}_{\text{SA}} \setminus \mathcal{S}_{\text{A}}$ in M5). See [`obr_decoder.md`](file:///home/abdelhalimyasser/Downloads/jssp-research/docs/algorithms/obr_decoder.md) and [`obr_machine_sequence_adapter.md`](file:///home/abdelhalimyasser/Downloads/jssp-research/docs/algorithms/obr_machine_sequence_adapter.md).
2. **QMC Mapping, Dimension Limits & Sizing Frozen (`D02`):** Confirmed dynamic remaining work $W_j(t)$, sequential inverse-CDF mapping over ascending `JobId`, Joe & Kuo (2008) `new-joe-kuo-6.21201` Sobol ($d_{\max} = 21,201$), Matoušek (1998) scrambling, baseline population $N = 64 = 2^6$, explicit contiguous prefix indices, and explicit $L \le 21,201$ restriction for standard Sobol. Sample sizes $N_S \in \{64, 32, 16\}$ are powers of two ($2^6, 2^5, 2^4$), whereas `A13`'s $48$-point Sobol prefix ($i=0\dots 47$) is explicitly labelled an empirical mixed-QMC configuration without standard dyadic digital-net balance guarantees. See [`qmc_mapping.md`](file:///home/abdelhalimyasser/Downloads/jssp-research/docs/algorithms/qmc_mapping.md).
3. **Neighborhoods & NNS/Approximate Evaluator Disposition (`D03`):** Confirmed primary citations for `N5`–`N8`; classified Gui et al. (2023)'s objective wording as internally inconsistent and deferred `NNS` to a paired-start, equal-budget supplementary Tabu Search ablation (`B15-NNS` vs. verified `B15 = Tabu Search + N7`); rejected Ba et al. (2026) approximate evaluator for primary objective accounting. See [`n5_n6_n7_n8_definitions.md`](file:///home/abdelhalimyasser/Downloads/jssp-research/docs/algorithms/n5_n6_n7_n8_definitions.md).
4. **Configuration Count Reconciled: 67 $\to$ 63 (`D04`):** Formally **removed standalone Metropolis–Hastings (`B25`–`B28`)** because without Hastings correction it duplicates fixed-temperature Metropolis/SA (`B21`–`B24`), while with exact Hastings correction it requires reverse-neighborhood enumeration and rejects moves with $q_k(\sigma \mid \sigma') = 0$ (global non-irreducibility and `N6`–`N8` $\to$ `N5` collapse are downgraded to unverified structural hypotheses; `D04-A`: `REJECTED`). Retained `B29`–`B32` as **Metropolis-filtered ILS perturbation followed by deterministic local search**, operating on OBR states $\pi \in \Pi^n_m$ with $E(\pi) = C_{\max}(\mathrm{Decode}(\pi))$ via the 3-cycle distinct-job OBR perturbation kernel ($D_{3\text{cycle}} = 2\sum_{a<b<c} m_a m_b m_c$) (`D04-B`: `ACCEPTED`). Active configuration total is now **63** (**24** Phase A + **28** Phase B + **11** Phase C). Original IDs `B29`–`B32` are preserved to avoid renumbering. See [`mh_vs_sa.md`](file:///home/abdelhalimyasser/Downloads/jssp-research/docs/algorithms/mh_vs_sa.md).
5. **Phase B Parameters & Fairness Frozen (`D05`):** Frozen scan ordering, Tabu dynamic tenure, ILS perturbation, SA geometric cooling, equal evaluation budgets ($100L$ screening, $500L$ confirmatory), and local-optimum reporting constraints. See [`phase_b.md`](file:///home/abdelhalimyasser/Downloads/jssp-research/docs/experiments/phase_b.md).
6. **Dual-View Gate Model & Adaptive Rule Specified (`D06`):** Selected **Model A (Pair-Level Selection Gate)** with $N_{\text{pairs}} = 32$ ($64$ genotypes), fixed weight $p$ on the Lamarckian view, and a deterministic EWMA adaptive controller (`C06`) contrasted against El-Mihoub et al. (2021). See [`dual_view.md`](file:///home/abdelhalimyasser/Downloads/jssp-research/docs/algorithms/dual_view.md).
7. **GA Baseline, Splits, Seeds & Track 4/5 Policy Frozen (`D07`):** Specified Classical GA baseline ($N=64$, binary tournament, POX $p_c=0.85$, swap/insert mutation $p_m=0.15$, elitism $E=2$), 20 frozen master seeds with 9 isolated PRNG substreams, 20%/80% stratified Track 1 split (`48` dev / `194` test), `160`-instance Track 3 sample (`40` dev / `120` test), and explicit deferral of Tracks 4 and 5. See [`phase_c.md`](file:///home/abdelhalimyasser/Downloads/jssp-research/docs/experiments/phase_c.md) and [`fairness_protocol.md`](file:///home/abdelhalimyasser/Downloads/jssp-research/docs/experiments/fairness_protocol.md).

---

## 2. Reconciled Configuration Matrix (63 Active Primary Configurations)

| Phase | Scope | Active Count | IDs Retained | Pruned / Supplementary |
|---|---|---:|---|---|
| **Phase A** | Initial Population Generation | **24** | `A01`–`A24` | None |
| **Phase B** | Local Search & Metaheuristics | **28** | `B01`–`B24`, `B29`–`B32` | **`B25`–`B28` pruned** (`D04-A`); supplementary `B15-NNS` isolated outside primary count (`D03-B`) |
| **Phase C** | Memetic & Dual-View GAs | **11** | `C01`–`C11` | None |
| **Total** | **Primary Factorial Scope** | **63** | `A01`–`A24`, `B01`–`B24`, `B29`–`B32`, `C01`–`C11` | **-4** relative to M0 67-config baseline |

### 2.1 Phase A Matrix (`A01`–`A24`, 24 Configurations)
- `A01`: Uniform Random ($\alpha = 0$)
- `A02`: Deterministic MWR (lowest `JobId` tie-break)
- `A03`, `A15`, `A16`: Monte Carlo + MWR ($\alpha = 1.0, 0.5, 2.0$)
- `A04`, `A05`: Halton, Scrambled Halton ($\alpha = 0$)
- `A06`, `A07`: Sobol, Scrambled Sobol ($\alpha = 0$, $N = 64 = 2^6$)
- `A08`, `A17`, `A18`: Halton + MWR ($\alpha = 1.0, 0.5, 2.0$)
- `A09`, `A19`, `A20`: Scrambled Halton + MWR ($\alpha = 1.0, 0.5, 2.0$)
- `A10`, `A21`, `A22`: Sobol + MWR ($\alpha = 1.0, 0.5, 2.0$)
- `A11`, `A23`, `A24`: Scrambled Sobol + MWR ($\alpha = 1.0, 0.5, 2.0$)
- `A12`, `A13`, `A14`: Mixed Scrambled Halton ($i=1\dots N_H$) / Scrambled Sobol ($i=0\dots N_S-1$) (`32:32` $2^5$ Sobol net, `16:48` empirical non-$2^m$ Sobol prefix without standard balance guarantee, `48:16` $2^4$ Sobol net)

### 2.2 Phase B Matrix (`B01`–`B24`, `B29`–`B32`, 28 Active Configurations)

| Strategy | N5 | N6 | N7 | N8 | Status |
|---|:---:|:---:|:---:|:---:|---|
| Steepest Descent | `B01` | `B02` | `B03` | `B04` | Active |
| First Improvement | `B05` | `B06` | `B07` | `B08` | Active |
| Random Improvement | `B09` | `B10` | `B11` | `B12` | Active |
| Tabu Search | `B13` | `B14` | `B15` | `B16` | Active *(+ paired supplementary `B15-NNS` vs. `B15`)* |
| Iterated Local Search (ILS) | `B17` | `B18` | `B19` | `B20` | Active |
| Simulated Annealing (SA) | `B21` | `B22` | `B23` | `B24` | Active |
| Standalone Metropolis–Hastings | ~~`B25`~~ | ~~`B26`~~ | ~~`B27`~~ | ~~`B28`~~ | **PRUNED / REJECTED (`D04-A`)** |
| Metropolis-Filtered ILS Perturbation + LS | `B29` | `B30` | `B31` | `B32` | Active (`D04-B`) |

### 2.3 Phase C Matrix (`C01`–`C11`, 11 Configurations)
- `C01`: Classical GA baseline ($N = 64$ genotypes)
- `C02`: Lamarckian GA (LGA)
- `C03`: Baldwinian GA (BGA)
- `C04`: Partial Lamarckian GA (PLGA, 50% deterministic top-odd rank assignment)
- `C05`, `C07`, `C08`: Dual-View Model A Fixed Gate ($p = 0.50, 0.25, 0.75$ on Lamarckian view; $N_{\text{pairs}} = 32$)
- `C06`: Dual-View Model A Adaptive Gate (rule-based EWMA on $\Delta_{\text{perf}}(t)$ and $\Delta_{\text{div}}(t)$)
- `C09`: Dual-View Adaptive Gate ablation without diversity signal ($w_{\text{div}} = 0$)
- `C10`: Unpaired Dual-Population Control ($32$ LGA + $32$ BGA without shared $\mathrm{ID}_i$)
- `C11`: Dual-View Controlled Cross-View Migration (every $T_{\text{mig}} = 10$ gens, top $10\%$ pairs)

---

## 3. Reconciled Benchmark Allocation

| Track | Dataset | Total | Dev / Screening | Held-Out Confirmatory / Test | Status |
|---|---|---:|---:|---:|---|
| **Track 1** | Core Classical Rectangular (FT, LA, ABZ, ORB, SWV, YN, TA, DMU) | 242 | **48** | **194** | **ACTIVE (`ACCEPTED`)** |
| **Track 2a** | Large-TA Rectangular Extension (8 strata $< 1000\times 1000$) | 80 | — | 80 (shortlist scalability) | **ACTIVE (`ACCEPTED`)** |
| **Track 2b** | Large-TA Extreme Stress ($1000\times 1000$) | 10 | — | 10 (shortlist stress only; no standard Sobol) | **ACTIVE (`ACCEPTED`)** |
| **Track 3** | Strassl–Musliu Synthetic Sample (from 4,225) | 160 | **40** | **120** | **ACTIVE (`ACCEPTED`)** |
| **Track 4** | Da Col–Teppan Variable-Route / Re-entrant | 24 | — | — | **DEFERRED (`D07-C`)** |
| **Track 5** | BEL Industrial Extension | 20 | — | — | **DEFERRED (`D07-C`)** |

---

## 4. Cross-References to Detailed M1-B Specifications
- Algorithm specs: [`obr_decoder.md`](file:///home/abdelhalimyasser/Downloads/jssp-research/docs/algorithms/obr_decoder.md), [`qmc_mapping.md`](file:///home/abdelhalimyasser/Downloads/jssp-research/docs/algorithms/qmc_mapping.md), [`n5_n6_n7_n8_definitions.md`](file:///home/abdelhalimyasser/Downloads/jssp-research/docs/algorithms/n5_n6_n7_n8_definitions.md), [`mh_vs_sa.md`](file:///home/abdelhalimyasser/Downloads/jssp-research/docs/algorithms/mh_vs_sa.md), [`dual_view.md`](file:///home/abdelhalimyasser/Downloads/jssp-research/docs/algorithms/dual_view.md)
- Experiment protocols: [`phase_a.md`](file:///home/abdelhalimyasser/Downloads/jssp-research/docs/experiments/phase_a.md), [`phase_b.md`](file:///home/abdelhalimyasser/Downloads/jssp-research/docs/experiments/phase_b.md), [`phase_c.md`](file:///home/abdelhalimyasser/Downloads/jssp-research/docs/experiments/phase_c.md), [`fairness_protocol.md`](file:///home/abdelhalimyasser/Downloads/jssp-research/docs/experiments/fairness_protocol.md)
- Bibliography: [`jssp_related_work.bib`](file:///home/abdelhalimyasser/Downloads/jssp-research/paper/bibliography/jssp_related_work.bib)
