# Scheduling & Reproducibility Data Flow (M2–M4)

This document traces the authoritative end-to-end data flow from raw instance inputs and OBR chromosomes through M2 scheduling validation, M3 reproducibility tracking, and M4 disjunctive-graph critical-path analysis.

---

## 1. End-to-End Scheduling & Graph Analysis Pipeline

```text
  Raw Instance rows: jobs[j] = [(machine, duration), ...]
        │
        ▼
  JsspInstance::new(...)                     ── checks J>=1, M>=1, rectangularity, machine perms, p>0, sum(p)<=u64::MAX
        │
        ▼
  ObrChromosome::resolve_operations(&inst)   ── checks length == J*M, 0 <= j < J, count[j] == M
        │
        ▼
  decoder::decode_obr(&inst, &obr)           ── canonical Serial SGS (earliest feasible idle-gap insertion, u64 checked_add)
        │
        ▼
  Schedule { operations, makespan }          ── sorted deterministic per-job & per-machine operation views
        │
        ├──► feasibility::validate_schedule(&inst, &schedule)
        │       └── checks operation coverage, duration arithmetic, job precedence, machine exclusivity, C_max
        │
        ├──► objective::evaluate_makespan(&inst, &schedule)
        │
        └──► M4: graph::DisjunctiveGraph::from_schedule(&inst, &schedule)
                ├── invokes feasibility::validate_schedule(&inst, &schedule) (no duplicate validation logic)
                ├── extracts chronological schedule.machine_sequences()
                ├── creates Source(0), Operation(1..=L), Sink(L+1) with JobPrecedence, MachineOrder, Source, Sink arcs
                └── verifies acyclicity via Kahn topological_sort() (BinaryHeap<Reverse<VertexId>>)
                        │
                        ▼
                    graph::analyze_disjunctive_graph(&graph)
                        ├── forward topological DP  ──► head[u], maximizing_predecessors, representative_predecessor
                        ├── reverse topological DP  ──► tail[u], maximizing_successors, representative_successor
                        ├── makespan equivalence    ──► verifies Cmax_graph == Cmax_schedule == max_u(head[u] + p[u] + tail[u])
                        ├── operation & arc slack   ──► slack[u] == 0, is_head_tight && v_critical -> CriticalSubgraph
                        ├── deterministic path      ──► RepresentativeCriticalPath (lowest OperationId backtracking from Sink)
                        └── machine-order scan      ──► CriticalBlock[] (maximal contiguous critical MachineOrder runs, len >= 2)
```

---

## 2. Negative / Malformed Validation Paths

1. **Invalid OBR / Instance:** Rejected cleanly in `JsspInstance::new` or `ObrChromosome::validate` before schedule decoding (`TimeOverflow` $\to$ `OverflowError`; domain violations $\to$ `ValueError`).
2. **Invalid Schedule:** Rejected inside `DisjunctiveGraph::from_schedule` via `validate_schedule`.
3. **Malformed Cyclic Orientation:** `DisjunctiveGraph::from_machine_sequences_unchecked` permits constructing a cyclic orientation for diagnostics; calling `topological_sort()` or `feasibility::check_disjunctive_graph_acyclicity` returns `JsspError::GraphCycleDetected { unvisited_vertices, cycle_operations }`.
4. **Non-Semi-Active Schedule Makespan Discrepancy:** Feasible right-shifted schedules whose idle delays push `schedule.makespan() > graph_makespan` are rejected cleanly by `analyze_disjunctive_graph` via `JsspError::GraphMakespanMismatch`.
5. **Reproducibility Boundary:** Every stochastic seed stream comes from M3's `rng::streams` (`ChaCha20Rng` + binary SHA-256) and immutable `RunManifest` metadata (`benchmark_data_flow.md`, `rng_architecture.md`, `run_manifest.md`). M4 graph analysis itself is 100% deterministic with zero RNG draws.
