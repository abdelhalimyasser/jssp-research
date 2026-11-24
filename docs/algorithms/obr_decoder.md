# OBR Representation, Serial SGS Decoder & Disjunctive-Graph Adapter (`D01`)

**Decision IDs:** `D01-A` (`ACCEPTED`), `D01-B` (`PROVISIONAL`)
**Scope:** Canonical genotype representation, schedule generation decoder, deterministic invariants, and bidirectional OBR $\leftrightarrow$ disjunctive-graph adapter for Phases A, B, and C.

---

## 1. Operation-Based Representation (OBR)

Let an instance $\mathcal{I} = (J, M, O, \mu, p)$ contain $n = |J|$ jobs, $m = |M|$ machines, and operation set $O = \{O_{j,k} \mid j \in \{0,\dots,n-1\},\, k \in \{0,\dots,m_j-1\}\}$ of total size $L = \sum_{j=0}^{n-1} m_j$ ($L = n m$ for rectangular classical JSSP).

- **Definition (Gen, Tsujimura & Kubota, 1994, DOI `10.1109/ICSMC.1994.400072`; Bierwirth, 1995, DOI `10.1007/BF01719250`):**
  An OBR chromosome is a multiset permutation $\pi = (\pi_0, \pi_1, \dots, \pi_{L-1}) \in \{0,\dots,n-1\}^L$ such that every job $j$ satisfies exact multiplicity:
  $$|\{t \in \{0,\dots,L-1\} : \pi_t = j\}| = m_j$$
- **Operation Indexing:** If $c_t = |\{s < t : \pi_s = \pi_t\}|$, then gene position $t$ uniquely denotes operation $O_{\pi_t, c_t}$.
- **Properties & Literature Trade-offs:**
  - *Advantage:* Every chromosome satisfying job multiplicity respects intra-job technological precedence ($O_{j,k} \prec O_{j,k+1}$) by construction, eliminating precedence-infeasible permutations and enabling valid crossover/mutation (GOX/POX) without repair.
  - *Limitation (Redundancy / Symmetry):* The mapping $\pi \mapsto S$ is many-to-one. Distinct chromosomes sharing the same topological ordering on machines decode to the identical schedule (and under gap insertion, even more distinct OBR strings map to the same active schedule).
- **Mandatory Validation Invariants:** Before decoding, `jssp-core` validates `pi.len() == L`, `pi[t] < n` for all $t$, and `count[j] == m_j` for all $j \in J$.

---

## 2. Schedule Space Comparison & Decoder Selection (`D01-A`: `ACCEPTED`)

Following Giffler & Thompson (1960, DOI `10.1287/opre.8.4.487`) and Sprecher, Kolisch & Drexl (1995, DOI `10.1016/0377-2217(93)E0294-8`, Section 2):

1. **Semi-Active Schedules ($\mathcal{S}_{\text{SA}}$):** No operation can be started earlier without changing the processing order on any machine. Every optimal schedule has a semi-active counterpart, so $\mathcal{S}_{\text{SA}}$ always contains an optimum.
2. **Active Schedules ($\mathcal{S}_{\text{A}} \subset \mathcal{S}_{\text{SA}}$):** No operation can be started earlier without delaying at least one other operation or violating precedence/machine constraints (equivalently: no admissible idle gap precedes any operation). By Giffler & Thompson (1960) Theorem 1 and Sprecher et al. (1995), **every makespan-optimal schedule for regular (non-decreasing) objectives such as $C_{\max}$ has an active counterpart**: $\min_{S \in \mathcal{S}_{\text{A}}} C_{\max}(S) = \min_{S \in \mathcal{S}_{\text{SA}}} C_{\max}(S)$.
3. **Non-Delay Schedules ($\mathcal{S}_{\text{ND}} \subset \mathcal{S}_{\text{A}}$):** No machine is kept idle when at least one eligible operation is available to start. **Non-delay schedules can exclude all optimal $C_{\max}$ schedules** (classical counterexamples exist even for $2 \times 2$ / $3 \times 3$ instances).

### Evaluation of Five Candidate Decoding Schemes

| Candidate Scheme | Schedule Class Generated | Preserves Global Optimum? | Behavior on OBR | Decision |
|---|---|---|---|---|
| 1. Append-only list scheduling | Semi-active ($\mathcal{S}_{\text{SA}}$) | Yes, only in full space | Appends to machine tail $M_{\text{last}}[r]$; misses earlier idle intervals, degrading fitness and search efficiency. | **REJECTED** |
| 2. Generic Semi-Active decoding | Semi-active ($\mathcal{S}_{\text{SA}}$) | Yes | Identical to append-only for left-to-right OBR precedence decoding. | **REJECTED** |
| 3. **Serial SGS with earliest feasible idle-gap insertion** | **Active ($\mathcal{S}_{\text{A}}$)** | **Yes** | Scans OBR left-to-right; inserts $O_{j,k}$ into the earliest feasible machine idle gap $[a, b]$ satisfying $t = \max(C_{j,k-1}, a)$ and $t + p_{j,k} \le b$. | **`ACCEPTED`** |
| 4. Giffler–Thompson Priority Dispatching | Active ($\mathcal{S}_{\text{A}}$) | Yes | Requires conflict-set arbitration using dynamic priority ties rather than pure chromosome order. | **REJECTED** *(as standalone decoder; Serial SGS preserves chromosome fidelity)* |
| 5. Non-Delay Schedule Generation | Non-delay ($\mathcal{S}_{\text{ND}}$) | **No** | Can prune all optimal $C_{\max}$ schedules by forbidding intentional machine idleness. | **REJECTED** |

### Proof Sketch: Serial SGS with Earliest Gap Insertion Generates Active Schedules
*Claim:* Every schedule $S$ produced by scanning a valid OBR $\pi$ left-to-right and placing each operation $o = O_{j,k}$ at the earliest feasible time $S_o$ into machine $\mu(o)$'s ordered interval list is **active**.
*Proof:* Suppose for contradiction that $S$ is not active. Then there exists at least one operation $o^*$ in $S$ that can be left-shifted strictly earlier to $S'_{o^*} < S_{o^*}$ without altering the start times of any other operation and without violating job precedence or machine exclusivity. Let $o^*$ be placed at step $t$ of the Serial SGS scan. At step $t$, its job predecessor $O_{j,k-1}$ has already been scheduled at final completion $C_{j,k-1}$ (because $\pi$ has $k$ occurrences of $j$ before step $t$), and all operations already placed on machine $r = \mu(o^*)$ occupy fixed intervals strictly a subset of $S$'s intervals on $r$. Because $S'_{o^*} \ge C_{j,k-1}$ and $[S'_{o^*}, S'_{o^*} + p_{o^*})$ does not overlap any interval in $S$ on machine $r$, it cannot overlap any already-placed interval at step $t$. Hence $[S'_{o^*}, S'_{o^*} + p_{o^*})$ fits inside an earlier feasible gap or before the first interval at step $t$, contradicting that the decoder chose the **earliest** feasible start time $S_{o^*} > S'_{o^*}$. Therefore $S \in \mathcal{S}_{\text{A}}$. $\square$

---

## 3. Deterministic Implementation Rules & Pseudocode

1. **Integer Time Representation:** All processing times, start times, completion times, and $C_{\max}$ use `u64` (`Time = u64`). Additions use `checked_add`, returning `JsspError::TimeOverflow` on overflow (never wrapping). Zero-duration operations ($p_o = 0$) are explicitly supported as instantaneous transitions (`[t, t)` does not block machine intervals of nonzero duration, though classical benchmarks satisfy $p_o \ge 1$).
2. **Machine Idle-Gap Search Order:** For each machine $r \in \{0,\dots,m-1\}$, maintain a sorted list `intervals[r]` of scheduled pairs `(start, end, op_id)` ordered by `(start, end)`. Given `job_ready = C_{j,k-1}` (or `0` if `k == 0`):
   - Start `prev_end = 0`. Scan `(s_i, e_i, _)` in ascending order in `intervals[r]`.
   - Candidate start `cand = max(job_ready, prev_end)`.
   - If `cand + p_o <= s_i` (via checked addition), insert `(cand, cand + p_o, op_id)` immediately before `(s_i, e_i, _)` and stop.
   - Otherwise update `prev_end = max(prev_end, e_i)` and continue.
   - If no gap accommodates $p_o$, append `(max(job_ready, prev_end), max(job_ready, prev_end) + p_o, op_id)` at the tail.
3. **Equal-Time Tie-Breaking:** Ties never arise within a step (each gene picks one unique operation), and across steps `intervals[r]` preserves strict causal placement order: if an instantaneous ($p_o=0$) or equal-time case arises, tie-break by `(start, end, job_id, op_idx)`.
4. **Stable Operation Identifiers:** Every operation has immutable `OpId(u32)` assigned in canonical `(job_id, op_idx)` row-major order ($0 \dots L-1$). Never iterate `HashMap` or `HashSet` in decoding or tie-breaking.

```text
function decode_obr_serial_sgs(instance, pi):
    validate_obr(instance, pi)                      // len == L, 0 <= j < n, count[j] == m_j
    job_next_op = [0; n]                            // next operation index k per job
    job_ready   = [0; n]                            // C_{j, k-1}
    machine_seq = [[] for _ in 0..m]                // sorted list of (start, end, op_id)
    starts = [0; L]; ends = [0; L]

    for j in pi:                                    // strict left-to-right scan
        k = job_next_op[j]
        op = instance.op_id(j, k)
        r  = instance.machine(op)
        p  = instance.proc_time(op)
        ready = job_ready[j]

        prev_end = 0
        placed_idx = machine_seq[r].len()
        cand_start = ready

        for idx, (s_i, e_i, _) in enumerate(machine_seq[r]):
            cand = max(ready, prev_end)
            if cand.checked_add(p) <= s_i:
                cand_start = cand
                placed_idx = idx
                break
            prev_end = max(prev_end, e_i)
        else:
            cand_start = max(ready, prev_end)

        cand_end = cand_start.checked_add(p)
        machine_seq[r].insert(placed_idx, (cand_start, cand_end, op))
        starts[op] = cand_start
        ends[op]   = cand_end
        job_ready[j]   = cand_end
        job_next_op[j] = k + 1

    cmax = max(ends[o] for o in 0..L)               // 0 if L == 0
    return Schedule(starts, ends, machine_seq, cmax)
```

---

## 4. Worked $3 \times 3$ Example (Demonstrating Gap Insertion & Round-Trip)

Consider the $3 \times 3$ instance from Xie et al. (2023, Table 1, 0-indexed jobs $J_0, J_1, J_2$ and machines $M_0, M_1, M_2$):
- $J_0$: $(M_0, 2) \to (M_1, 1) \to (M_2, 3)$
- $J_1$: $(M_0, 1) \to (M_2, 2) \to (M_1, 2)$
- $J_2$: $(M_1, 5) \to (M_0, 2) \to (M_2, 1)$

Decode chromosome $\pi = [1, 2, 2, 2, 1, 0, 0, 0, 1]$ using `decode_obr_serial_sgs`:

| Step $t$ | Gene $j$ | Op | $(M_r, p)$ | `ready` | `intervals[r]` before | Chosen Interval | Mechanism |
|---:|---:|---|---|---:|---|---|---|
| 0 | `1` | $O_{1,0}$ | $(M_0, 1)$ | 0 | `[]` | `[0, 1]` | Machine tail |
| 1 | `2` | $O_{2,0}$ | $(M_1, 5)$ | 0 | `[]` | `[0, 5]` | Machine tail |
| 2 | `2` | $O_{2,1}$ | $(M_0, 2)$ | 5 | `[(0, 1, O10)]` | `[5, 7]` | Machine tail (`max(5,1)=5`) |
| 3 | `2` | $O_{2,2}$ | $(M_2, 1)$ | 7 | `[]` | `[7, 8]` | Machine tail (`ready=7`) |
| 4 | `1` | $O_{1,1}$ | $(M_2, 2)$ | 1 | `[(7, 8, O22)]` | **`[1, 3]`** | **Idle-gap insertion** (`max(1,0)+2 = 3 <= 7`) |
| 5 | `0` | $O_{0,0}$ | $(M_0, 2)$ | 0 | `[(0, 1, O10), (5, 7, O21)]` | **`[1, 3]`** | **Idle-gap insertion** (`max(0,1)+2 = 3 <= 5`) |
| 6 | `0` | $O_{0,1}$ | $(M_1, 1)$ | 3 | `[(0, 5, O20)]` | `[5, 6]` | Tail (`max(3,0)+1=4 > 0`, cannot precede `O20`) |
| 7 | `0` | $O_{0,2}$ | $(M_2, 3)$ | 6 | `[(1, 3, O11), (7, 8, O22)]` | `[8, 11]` | Tail (`max(6,3)+3=9 > 7`, misses gap `[3,7]`) |
| 8 | `1` | $O_{1,2}$ | $(M_1, 2)$ | 3 | `[(0, 5, O20), (5, 6, O01)]` | `[6, 8]` | Tail (`max(3,6)=6`) |

- **Resulting active schedule:** $C_{\max} = \max(11, 8, 8) = 11$ (note: pure append-only without gap insertion places $O_{1,1}$ at `[8, 10]`, $O_{0,0}$ at `[7, 9]`, $O_{0,1}$ at `[9, 10]`, $O_{0,2}$ at `[10, 13]`, and $O_{1,2}$ at `[10, 12]`, giving $C_{\max} = 13$).
- **Resulting machine orders:** $M_0: [O_{1,0}, O_{0,0}, O_{2,1}]$, $M_1: [O_{2,0}, O_{0,1}, O_{1,2}]$, $M_2: [O_{1,1}, O_{2,2}, O_{0,2}]$.

---

## 5. Critical Representation Adapter: OBR $\leftrightarrow$ Disjunctive Graph (`D01-B`: `PROVISIONAL`)

Neighborhoods `N5`–`N8` mutate **machine operation sequences** $\sigma = (\sigma_0, \dots, \sigma_{m-1})$ on an acyclic disjunctive graph $D = (O \cup \{0, \tilde{n}\}, A, E(\sigma))$.

### 5.1 Forward Conversion (`OBR -> Schedule / Disjunctive Graph`)
1. Run `decode_obr_serial_sgs(instance, pi)` to obtain `Schedule`.
2. Extract per-machine sequences `sigma[r] = [op_id for (_, _, op_id) in machine_seq[r]]`.
3. Construct directed machine disjunctive arcs $E(\sigma) = \{ (\sigma_{r,i}, \sigma_{r,i+1}) \}$. Because `Schedule` is valid and non-overlapping, $D(\sigma)$ is guaranteed acyclic.

### 5.2 Inverse Conversion (`Modified Disjunctive Graph -> Canonical OBR`)
After a neighborhood move mutates $\sigma \to \sigma'$ and validates acyclicity (Kahn's topological sort; cyclic graphs are immediately rejected):
1. Compute semi-active longest-path start times $r_o = L(0, o)$ for every $o \in O$ on the acyclic DAG $D(\sigma')$.
2. Sort all $L$ operations deterministically by key `(r_o, r_o + p_o, job_id, op_idx)` — this key is strictly consistent with both job conjunctive arcs (if $u \prec_{\text{job}} v$, $r_v \ge r_u + p_u \implies r_v > r_u$ for $p_u > 0$, or `op_idx` breaks ties when $p_u = 0$) and machine disjunctive arcs (if $u \prec_{\text{mach}} v$, $r_v \ge r_u + p_u$).
3. Extract job IDs in sorted order to form $\pi_{\text{canon}}$.
4. Re-decode `S_rt = decode_obr_serial_sgs(instance, pi_canon)` and assert:
   - `S_rt` is valid and feasible.
   - `S_rt.cmax <= cmax_graph` (guaranteed because `S_rt` is the active left-shift of $D(\sigma')$).
   - **Telemetry:** Track whether `S_rt.machine_seq == sigma'` (exact machine-order round-trip) or whether Serial SGS performed an active left-shift into an earlier gap (`adapter_left_shift_count += 1`).

**Worked Inverse Round-Trip Check:** From Section 4's schedule, sorting $(r_o, r_o+p_o, j, k)$ gives:
$(0,1,1,0), (0,5,2,0), (1,3,0,0), (1,3,1,1), (5,6,0,1), (5,7,2,1), (6,8,1,2), (7,8,2,2), (8,11,0,2)$
$\implies \pi_{\text{canon}} = [1, 2, 0, 1, 0, 2, 1, 2, 0]$, which re-decodes under `decode_obr_serial_sgs` to the **identical** start times, completion times, machine sequences, and $C_{\max} = 11$.

**Ambiguity & Non-Uniqueness Analysis:**
- *Active schedules:* Every active schedule is a fixed point of `(r_o, r_o+p_o, job_id)` re-encoding + Serial SGS decoding, preserving exact machine sequences.
- *Semi-active (non-active) post-move graphs:* If an `N5`–`N8` move unlocks an earlier machine idle gap on machine $r$, Serial SGS places the eligible operation into that gap, altering machine order on $r$ relative to $\sigma'$ while strictly improving or preserving $C_{\max}$.
- *Policy (`D01-B`, `PROVISIONAL`):* Retain the Serial SGS active schedule (never reject a strictly better active left-shift), log `adapter_left_shift_count`, and run dedicated M2/M5 tests to quantify left-shift frequency before confirmatory Phase B runs.

---

## 6. Proposed M2 Correctness Tests
1. **Multiplicity & Bounds Property Tests:** Any invalid length, out-of-range job ID, or wrong multiplicity returns `Err(JsspError::InvalidObr)`.
2. **Active-Schedule Property Test:** For every decoded schedule, verify no operation $o$ can be moved strictly earlier without overlapping another operation on $\mu(o)$ or violating $S_{j,k} \ge C_{j,k-1}$.
3. **Hand-Checked Fixture Test:** Verify the $3 \times 3$ gap-insertion example (`[1,2,2,2,1,0,0,0,1] -> Cmax = 11`) and confirm `O_{1,1}` lands at `[1, 3]` rather than `[8, 10]`.
4. **Independent Feasibility & Makespan Cross-Check:** Assert precedence, machine interval disjointness, and $C_{\max} = \max_o C_o$.
5. **Adapter Round-Trip Property Test:** Verify `S_rt.cmax <= cmax_graph` across random feasible disjunctive graphs and exact machine-order equality on all active schedules.

---

## 7. Primary References
- B. Giffler and G. L. Thompson, "Algorithms for Solving Production-Scheduling Problems," *Operations Research*, vol. 8, no. 4, pp. 487–503, 1960. DOI: `10.1287/opre.8.4.487`
- M. Gen, Y. Tsujimura, and E. Kubota, "Solving job-shop scheduling problems by genetic algorithm," *Proc. IEEE Int. Conf. Systems, Man and Cybernetics*, vol. 2, pp. 1577–1582, 1994. DOI: `10.1109/ICSMC.1994.400072`
- C. Bierwirth, "A generalized permutation approach to job shop scheduling with genetic algorithms," *OR Spektrum*, vol. 17, no. 2–3, pp. 87–92, 1995. DOI: `10.1007/BF01719250`
- A. Sprecher, R. Kolisch, and A. Drexl, "Semi-active, active, and non-delay schedules for the resource-constrained project scheduling problem," *European Journal of Operational Research*, vol. 80, no. 1, pp. 94–102, 1995. DOI: `10.1016/0377-2217(93)E0294-8`

---

## 8. Milestone M2–M5 Implementation Status & Concrete API Mapping

As of Milestones M2–M5, Sections 1–5 of this specification are implemented in pure Rust (`crates/jssp-core`) and exposed through the PyO3 FFI bridge (`crates/jssp-python` / `jssp_research.native`). Disjunctive-graph critical-path/block analysis (M4) and the Section 5 inverse `D01-B` OBR $\leftrightarrow$ machine-sequence adapter with quantitative round-trip telemetry (M5) are verified in [`disjunctive_graph.md`](file:///home/abdelhalimyasser/Downloads/jssp-research/docs/algorithms/disjunctive_graph.md), [`critical_path_and_blocks.md`](file:///home/abdelhalimyasser/Downloads/jssp-research/docs/algorithms/critical_path_and_blocks.md), and [`obr_machine_sequence_adapter.md`](file:///home/abdelhalimyasser/Downloads/jssp-research/docs/algorithms/obr_machine_sequence_adapter.md). Neighborhood move generation (`N5`–`N8`) remains deferred to Milestone M6+.

*(Audit note on `D01-B` status: Milestone M5 proved 100% `Exact` fixed-point round-tripping across all active schedules generated by `decode_obr`, and established that non-active semi-active orientations project via left-shifts ($S'_o \le r_o$, `right_shifts == 0`, $C_{\max}' \le C_{\max}^{\text{graph}}$) with explicit audit counters `adapter_left_shift_count` and `machine_order_inversion_count`.)*

### Implemented Rust Components (`crates/jssp-core`)

- **`error::JsspError`:** Typed error enum covering instance validation (`EmptyJobs`, `EmptyMachines`, `DimensionOverflow`, `NonRectangularJobLength`, `InvalidMachineId`, `DuplicateMachineInJob`, `NonPositiveProcessingTime`, `TimeOverflow`), OBR checks (`InvalidChromosomeLength`, `InvalidJobId`, `InvalidJobMultiplicity`), and independent schedule feasibility (`EmptySchedule`, `MissingOperation`, `DuplicateOperation`, `UnexpectedOperation`, `OperationMachineMismatch`, `OperationProcessingTimeMismatch`, `ProcessingDurationMismatch`, `JobPrecedenceViolation`, `MachineOverlapViolation`, `MakespanMismatch`).
- **`model`:**
  - `JobId`, `MachineId`, `OperationIndex`, `OperationId` (with `flat_index` and `from_flat_index` row-major mapping).
  - `Operation` and `JsspInstance` (`JsspInstance::new`, `JsspInstance::from_matrices`). Consistent with Classical JSSP benchmarks, Milestone M2 enforces strict positivity `processing_time > 0` at construction and checks `u64` total work against overflow.
  - `ScheduledOperation`, `MachineSequence`, and `Schedule` (with deterministic `.job_operations`, `.machine_operations`, `.machine_sequences`, and `.job_sequences` sorted explicitly by `(start_time, completion_time, job, op_index)`).
- **`representation::ObrChromosome`:** Wraps `Vec<JobId>`, validates length ($J \times M$), index bounds ($0 \le j < J$), and exact job multiplicity ($M$), and resolves gene positions left-to-right via `resolve_operations`.
- **`decoder::{SerialInsertionDecoder, decode_obr, decode_obr_append_only}`:** Implements canonical Serial SGS earliest feasible idle-gap insertion (`decode_obr`) using checked `u64` arithmetic (`checked_add`), alongside `decode_obr_append_only` strictly as a testing reference demonstrating the semi-active vs. active gap-insertion contrast (reproducing $C_{\max} = 11$ vs. $13$ on the Section 4 Xie et al. Table 1 $3 \times 3$ fixture).
- **`feasibility::validate_schedule` & `objective::evaluate_makespan`:** Independently verifies operation coverage, instance metadata consistency, interval arithmetic, job precedence, pairwise machine interval exclusivity on $[S_a, C_a)$, and makespan recomputation without relying on internal decoder structures.
