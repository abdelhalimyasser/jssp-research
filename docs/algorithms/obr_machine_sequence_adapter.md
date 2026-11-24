# OBR $\leftrightarrow$ Machine-Sequence Adapter & Round-Trip Audit (`D01-B`, Milestone M5)

**Status:** Implemented & Verified in `crates/jssp-core/src/representation/adapter.rs` and exposed via PyO3 (`crates/jssp-python/src/graph_api.rs`).
**Decision Disposition (`D01-B`):** **`PROVISIONAL` *(Restricted Exactness Verified on $\mathcal{S}_{\text{A}}$)*** — Exact fixed-point round-tripping holds across 100% of active schedules produced by M2 Serial SGS (`decode_obr`), whereas semi-active non-active machine orientations ($\mathcal{S}_{\text{SA}} \setminus \mathcal{S}_{\text{A}}$) deterministically left-shift into earlier idle gaps rather than preserving non-active machine orders.

---

## 1. Purpose and Scope

In `jssp-research`, genetic/memetic algorithms evolve **Operation-Based Representation (OBR)** chromosomes decoded with the canonical M2 active Serial SGS (`decode_obr`), whereas future neighborhood structures (`N5`–`N8`) operate on oriented **machine-sequence graphs**. Milestone M5 implements and empirically evaluates the inverse bridge:

$$\text{Target Machine Sequences } \sigma = (\sigma_0, \dots, \sigma_{M-1}) \;\longrightarrow\; \text{Acyclic } D(\sigma) \;\xrightarrow{\text{head DP}}\; \text{OBR } \pi_{\text{canon}} \;\xrightarrow{\text{decode\_obr}}\; S' \text{ (+ Audit)}$$

No `N5`–`N8` moves, local search loops, or memetic inheritance are modified or implemented here; M2 Serial SGS (`decode_obr`) and M4 critical-path semantics remain untouched.

---

## 2. Validated Machine-Sequence Graph Construction

`DisjunctiveGraph::from_target_machine_sequences(instance, machine_sequences)` validates:
1. Exactly $M$ sequences with `seq.machine == MachineId(m)` and length $J$ (`InvalidDisjunctiveGraph`).
2. Every operation ID is within $0..J \times 0..M$ (`UnexpectedOperation`) and belongs to machine $m$ (`OperationMachineMismatch`).
3. Every operation appears at most once (`DuplicateOperation`) and all $L = J M$ operations are covered (`MissingOperation`).
4. Intra-job precedence arcs `O(j, k) -> O(j, k + 1)` and machine-order arcs are acyclic via `topological_sort()` (`GraphCycleDetected` with cycle witness on cycle).
5. Exact longest-path makespan $C_{\max}(D)$ is computed over topological order with checked `u64` arithmetic (`TimeOverflow`).

---

## 3. Canonical Topological OBR Encoding (`D01-B`)

Given an acyclic graph $D$ and M4 `CriticalPathAnalysis` head times $r_o = \mathrm{head}[o]$ (`r_o` excludes `p_o`):

1. Form the checked sorting key per operation $o = (j, k)$:
   $$K(o) = \big(r_o,\; r_o + p_o,\; j,\; k\big)$$
   using `checked_add` (`TimeOverflow` on overflow).
2. Sort all $L$ operations lexicographically ascending by $K(o)$.
3. Verify intra-job technological order: because intra-job arcs $O_{j,k} \to O_{j,k+1}$ enforce $r_{j,k+1} \ge r_{j,k} + p_{j,k} > r_{j,k}$ (for $p_{j,k} > 0$, with $k$ breaking any tie), every job's operations appear in strict order $k = 0, \dots, M-1$.
4. Emit gene `j` for each sorted operation to construct `ObrChromosome`, and validate via M2's single authoritative `ObrChromosome::validate` and `ObrChromosome::resolve_operations`.

---

## 4. Mathematical Round-Trip Properties & Exactness Conditions

### Theorem 1 — Pointwise Left-Shift Dominance ($S'_o \le r_o$) on Any Acyclic Orientation
For **every** valid acyclic target machine-sequence graph $D(\sigma)$ encoded via `encode_graph_to_obr` and decoded via M2 canonical `decode_obr`:
1. The generated OBR chromosome $\pi_{\text{canon}}$ is **always valid** ($L = JM$, exact multiplicity $M$, technological order preserved).
2. Let $S'$ be the schedule produced by `decode_obr(instance, pi_canon)`. Then for every operation $o \in O$:
   $$S'_o \le r_o, \qquad \mathrm{adapter\_right\_shift\_count} = 0, \qquad C_{\max}(S') \le C_{\max}(D(\sigma)).$$
*Proof sketch:* Induct along chromosome position in $\pi_{\text{canon}}$. Because operations are sorted by $(r_o, r_o + p_o, j, k)$, when step $t$ places operation $v$, both its job predecessor $JP[v]$ ($r_{JP[v]} + p_{JP[v]} \le r_v$) and every machine predecessor $q$ appearing earlier on $\mu(v)$ ($r_q < r_v \implies r_q + p_q \le r_v$) have already been scheduled at times $S'_q \le r_q$ with completions $C'_q \le r_q + p_q \le r_v$. Thus `[r_v, r_v + p_v)` is always free at or before $r_v$, forcing `decode_obr` to select $S'_v \le r_v$. $\square$

### Theorem 2 — Restricted Exactness (Fixed-Point Property on Active Schedules $\mathcal{S}_{\text{A}}$)
If the target orientation $\sigma$ is the machine sequence of an **active** schedule $S \in \mathcal{S}_{\text{A}}$ (such as any schedule returned by `decode_obr`), then no operation in $S$ admits an earlier gap without delaying another operation. Combining Theorem 1 ($S'_o \le r_o$) with no-gap minimality forces $S'_o = r_o$ and identical machine sequences $\sigma' = \sigma$ across all machines, achieving `RoundTripStatus::Exact` (0 left-shifts, 0 right-shifts, 0 inversions, 0 makespan diff).

### Failure Mode on Semi-Active Non-Active Orientations ($\mathcal{S}_{\text{SA}} \setminus \mathcal{S}_{\text{A}}$)
When $\sigma$ encodes a **semi-active non-active** orientation containing an interior idle gap wide enough to accommodate a later operation on the same machine, M2 earliest-gap insertion *strictly moves that later operation earlier into the idle gap*. In this case:
- `decode_obr` changes the machine sequence relative to $\sigma$ (`machine_order_inversion_count > 0`, `mismatched_machine_arc_count > 0`, `adapter_left_shift_count > 0`), yielding `RoundTripStatus::MachineOrderAndStartTimesChanged` (or `StartTimesChanged` if order happened to stay intact).
- Crucially, **no alternative topological OBR sorting key can restore $\sigma$ under M2 Serial SGS** whenever the semi-active orientation has no pre-image in the active decoder's image space.

---

## 5. Worked Illustrative Adversarial Fixture (§6 $2 \times 2$ Case)

- **Instance:** $J_0: (M_1, 5) \to (M_0, 4)$; $J_1: (M_0, 2) \to (M_1, 3)$.
- **Target machine sequences:** $M_0 = [O_{0,1}, O_{1,0}]$, $M_1 = [O_{0,0}, O_{1,1}]$.
- **Target graph heads & makespan:** $r_{0,0}=0$, $r_{0,1}=5$, $r_{1,0}=9$, $r_{1,1}=11$; $C_{\max}(D) = 14$.
- **Canonical OBR:** $\pi_{\text{canon}} = [0, 0, 1, 1]$ (valid OBR).
- **Decoded schedule (`decode_obr`):**
  - $O_{0,0}$ on $M_1$: $[0, 5]$
  - $O_{0,1}$ on $M_0$: $[5, 9]$ (creates idle gap $[0, 5]$ on $M_0$)
  - $O_{1,0}$ on $M_0$: inserts into $[0, 2] \subset [0, 5]$ **ahead** of $O_{0,1}$
  - $O_{1,1}$ on $M_1$: $[5, 8]$
  - **Decoded $C_{\max} = 9$**; $M_0 = [O_{1,0}, O_{0,1}]$, $M_1 = [O_{0,0}, O_{1,1}]$.
- **Audit Telemetry:** `status = "machine_order_and_start_times_changed"`, `adapter_left_shift_count = 2` ($O_{1,0}: 9 \to 0$, $O_{1,1}: 11 \to 5$), `adapter_right_shift_count = 0`, `adapter_unchanged_start_count = 2`, `machine_order_inversion_count = 1`, `mismatched_machine_arc_count = 1`, `signed_makespan_diff = -5`.
- **Exhaustive Pre-Image Proof ($\Pi^2_2$):** Checking all $\binom{4}{2} = 6$ valid OBR chromosomes confirms **all 6** decode to $M_0 = [O_{1,0}, O_{0,1}]$ (`5` give $C_{\max} = 9$; `[1,1,0,0]` flips $M_1$ to `[O_{1,1}, O_{0,0}]` with $C_{\max} = 14$). Zero chromosomes map to $M_0 = [O_{0,1}, O_{1,0}]$ under earliest-gap Serial SGS.

---

## 6. Adapter Audit Telemetry & Empirical Evaluation

Every round trip produces a deterministic `AdapterAuditReport` containing:
- 64-hex SHA-256 digests (`target_graph_hash`, `generated_obr_hash`, `decoded_schedule_hash`)
- Exact integer `target_graph_makespan`, `decoded_schedule_makespan`, and overflow-safe `i128` `signed_makespan_diff`
- `adapter_left_shift_count` ($S'_o < r_o$), `adapter_right_shift_count` ($S'_o > r_o$), `adapter_unchanged_start_count` ($S'_o == r_o$)
- `machine_order_inversion_count` (Kendall pairwise inversions per machine relative to target order) and `mismatched_machine_arc_count` (target consecutive pairs missing as consecutive pairs in decoded sequence)
- Typed `RoundTripStatus`: `Exact`, `StartTimesChanged`, `MachineOrderChanged`, `MachineOrderAndStartTimesChanged`, `InvalidTargetGraph`

### Observed Reproducible Preservation Rates ($J, M \in 2..6$, 200 Seeded Instances)
- **Category A (M2 Serial SGS active schedules):** **`200 / 200` (`100.0%`) `Exact`**, `0` left shifts, `0` right shifts, `0` machine inversions, `0` makespan difference *(corroborated across 200 additional seeded $1..6$ Rust cases and 35 Hypothesis cases)*.
- **Category B (Append-only semi-active orientations):** `53 / 200` (`26.5%`) were already active (`Exact`); `147 / 200` (`73.5%`) left-shifted (`1,140 / 3,248` ops = `35.1%`, `569` pairwise machine inversions, `0 / 200` right shifts, `signed_makespan_diff` $\in [-99, 0]$, improving makespan strictly on `124 / 200` = `62.0%` cases).

---

## 7. Architectural Classification & Memetic / Neighborhood Implications (§9)

Empirical and formal results jointly establish **Cases B & C**: exact round-tripping holds across the full domain of M2-compatible active schedules ($\mathcal{S}_{\text{A}}$), whereas valid OBR chromosomes produced for arbitrary non-active acyclic machine orientations project into active schedules that can alter target machine orderings (`left-shift projection`, never worsening makespan). This behavior is an inherent mathematical consequence of keeping the active Serial SGS decoder (`D01-A`) fixed and must **not** be masked by altering `decode_obr`.

Implications for upcoming milestones:
1. **`N5`–`N8` Neighborhood Search (M6+):** Post-move makespan evaluation of graph moves should inspect the graph/schedule directly in graph space when precise step-level machine-order navigation matters, while logging `adapter_left_shift_count` and `machine_order_inversion_count` whenever a move is written back to OBR (e.g., for tabu hashing or memetic inheritance). Notice that left-shift projections strictly improve or match makespan ($C_{\max}(S') \le C_{\max}(D(\sigma))$).
2. **Lamarckian GA (`C02`):** Writing an improved schedule back via `D01-B` yields an exact OBR fixed point matching the resulting active schedule.
3. **Baldwinian GA (`C03`):** Retains the original unmutated OBR chromosome while assigning locally improved makespan fitness; even if local search visited a semi-active intermediate before stopping, no genotype corruption occurs.
4. **Partial Lamarckian (`C04`) & Dual-View (`C05`–`C11`):** Only the Lamarckian view $G^L$ undergoes `D01-B` write-back while paired $G^B$ maintains unmodified OBR lineage; both views remain comparable in $\mathcal{S}_{\text{A}}$, and audit metrics expose any projection rate empirically.

---

## 8. Limitations

- Non-active semi-active machine sequences outside `decode_obr`'s active schedule image cannot preserve exact non-active machine orders under active Serial SGS without changing `D01-A` (which is forbidden because semi-active / non-active append-only decoding is strictly inferior for GA search quality).
- Accordingly, **`D01-B` remains `PROVISIONAL`** with restricted exactness formally documented over active schedules $\mathcal{S}_{\text{A}}$.
