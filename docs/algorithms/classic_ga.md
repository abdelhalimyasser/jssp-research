# Classical Genetic Algorithm Baseline (`C01`)

**Rust Implementation:** [`crates/jssp-core/src/genetic/`](file:///home/abdelhalimyasser/Downloads/jssp-research/crates/jssp-core/src/genetic/mod.rs)
**Decoder Semantics:** Canonical M2 Serial SGS with earliest feasible idle-gap insertion ([`obr_decoder.md`](file:///home/abdelhalimyasser/Downloads/jssp-research/docs/algorithms/obr_decoder.md))
**Protocol Reference:** [`phase_c.md`](file:///home/abdelhalimyasser/Downloads/jssp-research/docs/experiments/phase_c.md) §1 (`D07-A`)

---

## 1. Frozen Operator Specification (`C01`)

| Component | Frozen Specification | Named RNG Substream | Source Module |
|---|---|---|---|
| Population Size | $N = 64$ OBR genotypes | — | [`population.rs`](file:///home/abdelhalimyasser/Downloads/jssp-research/crates/jssp-core/src/genetic/population.rs) |
| Parent Selection | Binary tournament ($k = 2$), tie-break lower index | `"selection"` | [`selection.rs`](file:///home/abdelhalimyasser/Downloads/jssp-research/crates/jssp-core/src/genetic/selection.rs) |
| Crossover | Precedence Operation Crossover (POX), $p_c = 0.85$, non-empty $1 \le |J_1| \le n-1$ | `"crossover"` | [`crossover.rs`](file:///home/abdelhalimyasser/Downloads/jssp-research/crates/jssp-core/src/genetic/crossover.rs) |
| Mutation | Per-offspring $p_m = 0.15$; 50% distinct-job OBR Swap / 50% distinct-job OBR Insertion | `"mutation"` | [`mutation.rs`](file:///home/abdelhalimyasser/Downloads/jssp-research/crates/jssp-core/src/genetic/mutation.rs) |
| Elitism | Generational replacement preserving top $E = 2$ individuals sorted by `(makespan, index)` | — | [`elitism.rs`](file:///home/abdelhalimyasser/Downloads/jssp-research/crates/jssp-core/src/genetic/elitism.rs) |
| Evaluation Budget | Phase A size-scaled $B_{\text{GA}}(n, m) = \mathrm{clamp}(64 \lceil 2.5 n m \rceil, 6400, 64000)$ | — | [`counters.rs`](file:///home/abdelhalimyasser/Downloads/jssp-research/crates/jssp-core/src/telemetry/counters.rs) |

---

## 2. Budget Parity, Reused-Workspace Equivalence & Telemetry

- **Unified Objective Accounting:** Both initial `64` evaluations (`try_consume_init`) and every post-initialization offspring decode (`try_consume_search`) charge the single `EvalCounter`. Elites preserved unchanged across generations are not re-decoded.
- **Zero-Allocation Workspace Equivalence:** Inside evolutionary generations, `SerialDecoderWorkspace::decode_makespan_only` reuses pre-allocated job/machine scratch buffers executing the identical `find_earliest_gap_slot` routine as `decode_obr`. Both initial ($t = 0$) and final populations, plus the best-so-far solution, additionally run full M2 `Schedule` construction and independent `validate_schedule` checks.
- **Layer A vs. Layer B Telemetry (`run_metrics.rs`):** Separates initial population quality/diversity (`PopulationDiagnostics`, $O(NL)$ pairwise normalized OBR Hamming diversity, unique OBR count, unique canonical machine-sequence schedule count, 1D KS discrepancy) from downstream GA convergence (`best_makespan`, `best_found_eval`, `best_found_generation`, `anytime_cmax_auc`, and per-generation trajectory).
