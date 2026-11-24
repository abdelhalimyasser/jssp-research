# Classical Job-Shop Scheduling Research Framework (`jssp-research`)

> A reproducible Rust/Python research framework evaluating population initialization, disjunctive-graph neighborhood local search, metaheuristics, and memetic / Dual-View genetic algorithms for the **classical deterministic rectangular Job-Shop Scheduling Problem (JSSP)** minimizing makespan ($C_{\max}$).

- **Repository Status:** All three experimental phases (`63` active configurations across `118,298` runs) and the integrated pipeline validation (`P1`–`P5`, `3,200` runs) are complete on Track 1 benchmarks (`242` classical instances; `32`-instance paired `16` Development / `16` Held-Out Common Suite; `20` frozen SHA-256 seeds).
- **Primary Integrated Winner:** **`P3_FrozenA_FrozenB` (`A11` Scrambled Sobol + MWR ($\alpha = 1.0$) + `B21` Simulated Annealing + `N5`)** — Held-Out Median RPD `3.959%`, Best RPD `1.742%`, Friedman rank `1.25 / 5`.

---

## 1. Research Overview & Three Sequential Phases

All experiments share identical scheduling semantics implemented in pure Rust ([`crates/jssp-core`](file:///home/abdelhalimyasser/Downloads/jssp-research/crates/jssp-core)): Operation-Based Representation (OBR), deterministic active Serial Schedule Generation Scheme (SGS) with earliest-gap insertion (`D01-A`), independent feasibility validation, and exact $O(L)$ disjunctive-graph evaluation (`D03-C`). Python ([`src/jssp_research`](file:///home/abdelhalimyasser/Downloads/jssp-research/src/jssp_research)) handles TOML config validation, matrix construction, benchmark provenance/splits, statistical analysis, and reporting via the PyO3/Maturin bridge ([`crates/jssp-python`](file:///home/abdelhalimyasser/Downloads/jssp-research/crates/jssp-python)).

| Phase | Scope | Active Configs | Runs | Development Winner | Held-Out Performance (Med RPD / Best RPD) |
|---|---|---:|---:|---|---|
| **Phase A** | Initial Population Generation (`A01`–`A24`: PRNG, MWR, MC+MWR, Halton/Sobol, Scrambled QMC, Mixed QMC) | `24` | `74,778` (Layer A, 242 inst.) + `15,360` (Layer B `C01` GA, 32 inst.) | **`A11`**: Scrambled Sobol + MWR ($\alpha = 1.0$) | `8.252%` / `5.325%` (Dev: `7.422%` / `5.139%`) |
| **Phase B** | Neighborhoods (`N5`–`N8`) $\times$ Search Strategies (`SD`, `FI`, `RI`, `TS`, `ILS`, `SA`, `ILS+MH`; `B25`–`B28` pruned per `D04-A`) | `28` | `17,920` (matched `A11` starts, budget $100L$) | **`B21`**: Simulated Annealing (SA) + `N5` | `4.349%` / `2.025%` (Dev: `3.951%` / `1.975%`) |
| **Phase C** | Classical (`C01`), Single-View Memetic (`C02`–`C04`), Paired Dual-View (`C05`–`C09`), Unpaired Control (`C10`), Exploratory Migration (`C11`) | `11` | `7,040` (embedded LS: `steepest_descent + N5`, $b_{\text{LS}}=50$) | **`C02`**: Lamarckian GA (LGA) *(numerical Dev/HO leader)* | `7.845%` / `5.463%` (Dev: `7.570%` / `5.701%`) |
| **Final** | Integrated Pipeline Validation (`P1`–`P5`, `A01`/`A11`/`B21`/`C01`/`C02`) | `5` pipelines | `3,200` (equalized $B_{\text{GA}}$) | **`P3_FrozenA_FrozenB`** (`A11` + `B21`) | **`3.959%`** / **`1.742%`** (Dev: **`3.782%`** / **`2.156%`**) |

> **Important Distinction (`C02` vs `P3`, `B21` vs Embedded LS, `P5` 85/15 Allocation):**
> - Within Phase C (`C02`–`C11`), embedded memetic local search used **`steepest_descent + N5` capped at $b_{\text{LS}} = 50$ evaluations** (`docs/experiments/phase_c.md` §3) rather than `B21` Simulated Annealing (whose 50-probe $T_0$ calibration would consume the full `b_LS = 50` cap). Full `B21` (`SA + N5`) is preserved in **`P3_FrozenA_FrozenB`** and Stage 2 of `P5_Integrated_A_C_B`.
> - While `C02` is the numerical Phase C winner, `C02` vs `C01` did **not** achieve significance after Holm correction ($p_{\text{Holm}} = 0.477$ Held-Out), and **`P3_FrozenA_FrozenB` (`A11` + `B21`)** is the best integrated pipeline overall (`16/0/0` wins over `P2`, $p_{\text{Holm}} = 2.136 \times 10^{-4}$, $A_{12} = 0.863$). `P5` reflects only the tested `85%/15%` LGA/SA budget split.

---

## 2. Preserved Dual-View Contribution & Honest Empirical Findings (`C05`–`C11`)

The proposed **Model A Paired Dual-View Architecture** (`docs/algorithms/dual_view.md`, `crates/jssp-core/src/memetic/dual_view/`) maintains `32` persistent identity-paired individuals $I_i = (\mathrm{ID}_i, G_i^L, G_i^B)$ (`64` total genotypes matching single-view `N=64`) with strictly within-view POX crossover and a pair-level selection gate $S_i(t) = p_t S_i^L(t) + (1 - p_t) S_i^B(t)$. All seven Dual-View/Baldwinian control and ablation configurations (`C05`–`C11`) are preserved in full:

- `C05` (`p=0.50`), `C07` (`p=0.25`), `C08` (`p=0.75`): Fixed Pair-Level Gate.
- `C06`: Deterministic EWMA Adaptive Gate ($w_{\text{perf}}=0.70, w_{\text{div}}=0.30$).
- `C09`: Adaptive Gate diversity-signal ablation ($w_{\text{perf}}=1.00, w_{\text{div}}=0.00$).
- `C10`: Unpaired Dual-Population Control (`32 LGA + 32 BGA`, tests pairing vs. independent subpools).
- `C11`: Exploratory Cross-View Migration (`T_mig = 10` generations).

**Empirical Outcome & 10-Factor Diagnosis (`reports/phase_c_summary.md` §2–3):**
- Under the equal-budget protocol ($B_{\text{GA}}$), **the tested Dual-View variants did not outperform single-view Lamarckian GA (`C02`)**: `C02` beat `C05` and `C06` on all `16/16` Held-Out instances ($p_{\text{Holm}} = 1.831 \times 10^{-4}$), and unpaired `C10` (`8.844%` Held-Out Med RPD) beat paired `C05` (`9.740%`) and `C06` (`9.472%`).
- Within Dual-View, higher Lamarckian weight monotonically improved results (`C03` `11.701%` $\to$ `C07` `10.581%` $\to$ `C05` `9.740%` $\to$ `C06` `9.472%` $\to$ `C08` `9.221%`), with `C06` significantly outperforming `C03` (`16/0/0`, $p_{\text{Holm}} = 1.831 \times 10^{-4}$) and `C07` (`13/2/1`, $p_{\text{Holm}} = 5.662 \times 10^{-3}$). Removing the diversity term (`C06` vs `C09`, $p_{\text{Holm}} = 0.363$) or adding migration (`C11` vs `C06`, $p_{\text{Holm}} = 0.300$) was not statistically significant on Held-Out.
- Raw trajectory audit highlights concrete structural causes under $B_{\text{GA}}$: `steepest_descent + N5` consumes `96.9%`–`97.5%` of evaluations, leaving only `~12` Held-Out generations (`11.85`–`13.23`)—too short for `C02` diversity collapse (`0.831` final OBR diversity) or for `C06`'s EWMA gate (`p_T = 0.5172 ± 0.0185`) to adapt far from `0.50`, while Model A spends ~50% of evaluations on the slower Baldwinian branch.

---

## 3. Results, Provenance & Raw Artifacts

- **Reports:** [`reports/final_research_results.md`](file:///home/abdelhalimyasser/Downloads/jssp-research/reports/final_research_results.md), [`reports/phase_a_summary.md`](file:///home/abdelhalimyasser/Downloads/jssp-research/reports/phase_a_summary.md), [`reports/phase_b_summary.md`](file:///home/abdelhalimyasser/Downloads/jssp-research/reports/phase_b_summary.md), [`reports/phase_c_summary.md`](file:///home/abdelhalimyasser/Downloads/jssp-research/reports/phase_c_summary.md)
- **Frozen Winner Locks & Summary CSVs/JSONs:** Committed under [`results/`](file:///home/abdelhalimyasser/Downloads/jssp-research/results/) (`winner/`, `analysis/`, `manifests/`).
- **Large Raw Outputs & Benchmark Files:** Git-ignored raw JSONL run logs (`233 MiB`) and MIT-licensed `SchedulingLab/jsp-instances` Track 1 files (`1.3 MiB`) are published via **GitHub Release `v1.0.0-confirmatory`** alongside `SHA256SUMS.txt`. Track 1 benchmarks can also be re-fetched and checksum-verified via [`scripts/fetch_benchmarks.py`](file:///home/abdelhalimyasser/Downloads/jssp-research/scripts/fetch_benchmarks.py) and [`scripts/validate_benchmarks.py`](file:///home/abdelhalimyasser/Downloads/jssp-research/scripts/validate_benchmarks.py).
- **Provenance Transparency:** Historical manifests preserve exact `git.dirty = true` flags recorded at execution time (`results/*/manifests/*.json`); see [`reports/final_research_results.md`](file:///home/abdelhalimyasser/Downloads/jssp-research/reports/final_research_results.md) §5 for the forensic audit of each commit delta and bit-for-bit statistical regeneration from raw JSONL files.

---

## 4. Environment Setup, Tests & Reproduction

Requires Rust `1.100.0` (pinned in [`rust-toolchain.toml`](file:///home/abdelhalimyasser/Downloads/jssp-research/rust-toolchain.toml)), `uv >= 0.13.9`, and Python `3.15.2` (pinned in [`.python-version`](file:///home/abdelhalimyasser/Downloads/jssp-research/.python-version)).

```bash
# 1. Install locked Python environment and build the PyO3/Maturin extension (release for experiments)
uv lock --check
uv sync --locked --all-groups
uv run --locked maturin develop --release --manifest-path crates/jssp-python/Cargo.toml

# 2. Run the full Rust + Python validation suite
cargo fmt --all -- --check
cargo check --workspace --all-targets
cargo test --workspace
cargo clippy --workspace --all-targets --all-features -- -D warnings
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pytest

# 3. Optional: acquire Track 1 benchmarks & run experiment pipelines
uv run --locked python scripts/fetch_benchmarks.py
uv run --locked python scripts/validate_benchmarks.py
uv run --locked python scripts/run_phase_a_pipeline.py
uv run --locked python scripts/run_phase_b_pipeline.py
uv run --locked python scripts/run_phase_c_pipeline.py
uv run --locked python scripts/run_final_pipeline.py
```

---

## 5. Project Completion Checklist (Track 1 Confirmatory Scope)

- [x] M0–M5 Foundation: Research spec, workspace, OBR/Serial-SGS decoder, RNG streams, benchmark catalog, disjunctive graph (`O(L)`), OBR↔machine-sequence adapter (`37` Python tests, `51` Rust tests).
- [x] Phase A (`A01`–`A24`, `74,778` Layer A + `15,360` Tier 2 runs) & frozen winner `A11`.
- [x] Phase B (`28` active configs `B01`–`B24, B29`–`B32`, `B25`–`B28` pruned per `D04-A`, `17,920` runs) & frozen winner `B21`.
- [x] Phase C (`C01`–`C11` incl. Dual-View `C05`–`C09`, control `C10`, exploratory `C11`, `7,040` runs) & numerical winner `C02`.
- [x] Final Integrated Pipeline (`P1`–`P5`, `3,200` runs) & overall winner `P3_FrozenA_FrozenB`.
- [x] GitHub Actions CI workflow (`.github/workflows/ci.yml`), `LICENSE`, `CITATION.cff`, and Release `v1.0.0-confirmatory`.
- [ ] Optional Future Extensions (explicitly out of Track 1 classical scope): Track 2 Large-TA (`80`+`10` stress), Track 3 Strassl–Musliu synthetic sample, Track 4 variable-route, Track 5 BEL industrial instances.
