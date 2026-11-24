# Compact Oriented Disjunctive Graph Model (Milestone M4)

**Status:** Implemented & Verified in `crates/jssp-core/src/graph/mod.rs` and exposed via PyO3 (`crates/jssp-python/src/graph_api.rs`).
**Scope:** Deterministic disjunctive-graph representation, topological sorting, validated machine-sequence construction (M5), and directed cycle diagnostics from validated classical rectangular JSSP schedules. Neighborhood moves (`N5`–`N8`) remain deferred beyond M5.

---

## 1. Mathematical Model & Scope Distinction

Let $\mathcal{I} = (J, M, O, p)$ be a validated classical rectangular JSSP instance with $n = |J|$ jobs, $m = |M|$ machines, and $L = n m$ operations, and let $S$ be a schedule passing independent M2 feasibility validation (`crate::feasibility::validate_schedule`).

We construct a **compact oriented disjunctive graph**:
$$D(S) = (V, A_{\text{job}} \cup A_{\text{mach}}(\sigma) \cup A_{\text{src}} \cup A_{\text{sink}})$$

> [!IMPORTANT]
> **Compact Consecutive Orientation vs. Pairwise Disjunctive Edges:**
> $D(S)$ stores only consecutive directed machine-order arcs $(\sigma_{r,i}, \sigma_{r,i+1})$ induced by the chronological machine sequences $\sigma_r$ of $S$ (plus conjunctive intra-job arcs and `Source`/`Sink` boundary arcs). It does **not** explicitly materialize all $O(n^2)$ transitive pairwise edges between every pair of operations on a machine. Its purpose is exact oriented feasibility verification, longest-path (`head`/`tail`) DP, critical subgraph construction, and critical-block extraction in $O(V + E) = O(L)$ memory.

---

## 2. Vertices & Node-Weight Convention

$D(S)$ has exactly $V = L + 2$ vertices indexed by canonical `VertexId(usize)`:

| `VertexId` | `GraphVertexKind` | Assigned `machine` | Node Weight $p[u]$ |
|---|---|---:|---:|
| `VertexId(0)` (`SOURCE`) | `GraphVertexKind::Source` | `None` | $0$ |
| `VertexId(1 + j \cdot M + k)` | `GraphVertexKind::Operation(OperationId(j, k))` | `Some(machine)` | $p_{j,k} > 0$ |
| `VertexId(L + 1)` (`Sink`) | `GraphVertexKind::Sink` | `None` | $0$ |

- Operation vertex IDs reuse M2's canonical row-major `OperationId::flat_index(J, M)` mapping directly (`1 + flat_index`). There is no second operation identity system.
- Node-weight ordering `VertexId(a) < VertexId(b)` on operation vertices is strictly isomorphic to lexicographic `(job, op_index)` ordering on `OperationId`.

---

## 3. Directed Arcs & Construction Order

Arcs are typed via `ArcType` and inserted in four deterministic passes without duplicate `(from, to, arc_type)` entries:

1. **`ArcType::JobPrecedence { job }` (`j in 0..J`, `k in 0..(M - 1)`):**
   Directed intra-job conjunctive arcs $O_{j,k} \to O_{j,k+1}$.
2. **`ArcType::MachineOrder { machine }` (`m in 0..M`, consecutive chronological pairs in $\sigma_m$):**
   For exact validated schedule sequence $\sigma_m = [u_0, u_1, \dots, u_{n-1}]$, directed arcs $u_i \to u_{i+1}$. Machine ordering is never inferred from OBR gene positions.
3. **`ArcType::Source` (`u in 1..=L`):**
   Directed arc $\mathrm{Source} \to u$ iff $u$ has zero incoming `JobPrecedence` and zero incoming `MachineOrder` arcs.
4. **`ArcType::Sink` (`u in 1..=L`):**
   Directed arc $u \to \mathrm{Sink}$ iff $u$ has zero outgoing `JobPrecedence` and zero outgoing `MachineOrder` arcs.

In classical rectangular JSSP (where every job visits each machine exactly once), no two consecutive operations of the same job share a machine, so `JobPrecedence` and `MachineOrder` arc sets are disjoint. Every operation vertex has in-degree $\le 2$ and out-degree $\le 2$, keeping total arc count $|E| = O(L)$.

---

## 4. Input Validation, Acyclicity & Cycle Diagnostics

- **`DisjunctiveGraph::from_schedule(instance, schedule)`** calls `crate::feasibility::validate_schedule(instance, schedule)?` before constructing the graph, rejecting invalid schedules through M2's `JsspError` variants without duplicating feasibility code, and immediately verifies acyclicity via `topological_sort()?`.
- **`DisjunctiveGraph::from_target_machine_sequences(instance, seqs)`** *(added in M5)* validates machine sequence count/ownership/length (`InvalidDisjunctiveGraph`), operation bounds (`UnexpectedOperation`), machine assignment (`OperationMachineMismatch`), uniqueness (`DuplicateOperation`), and completeness (`MissingOperation`), verifies acyclicity via `topological_sort()?`, and populates the exact longest `Source`-to-`Sink` path makespan via checked `u64` arithmetic (`TimeOverflow`).
- **`DisjunctiveGraph::from_machine_sequences_unchecked(instance, seqs, makespan)`** validates machine coverage/ownership and constructs the adjacency lists without rejecting cycles at build time so that `topological_sort()` (and `feasibility::check_disjunctive_graph_acyclicity`) can detect malformed cyclic graphs and report concrete cycle witnesses in tests and diagnostics.
- **Deterministic Topological Sorting (Kahn + Min-Heap):**
  - Uses `std::collections::BinaryHeap<Reverse<VertexId>>` as the ready queue.
  - Because `VertexId` orders `Source (0) < Operation(0,0) < ... < Operation(J-1,M-1) < Sink (L+1)`, simultaneous ready operations break ties strictly by `(job, op_index)`, preserving `Source` first and `Sink` last.
  - If fewer than $V$ vertices are popped, `extract_cycle_witness` traces backward along minimum unvisited predecessors (`in_degree > 0`) to recover a concrete forward-oriented directed cycle `cycle_operations: Vec<(usize, usize)>`, returning `Err(JsspError::GraphCycleDetected { unvisited_vertices, cycle_operations })`.

---

## 5. Complexity & Determinism Summary

| Component | Data Structure | Time Complexity | Space Complexity | Notes |
|---|---|---|---|---|
| Graph construction | Flat vertex array + `Vec<Vec<ArcId>>` adjacency lists | $O(V + E)$ after schedule validation | $O(V + E) = O(L)$ | No `HashMap` iteration order exposed |
| Topological sort | Kahn + `BinaryHeap<Reverse<VertexId>>` | $O((V + E) \log V)$ | $O(V)$ | Deterministic `(job, op_index)` tie-breaking |
| Cycle witness extraction | Deterministic min-predecessor walk | $O(V + E)$ | $O(V)$ | Only executed on invalid cyclic input |

All Rust structures avoid unsafe code, external graph dependencies, and hash-order sensitivity.
