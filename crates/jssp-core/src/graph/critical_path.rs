//! Longest-path DP (head/tail times), critical operations, critical arcs, critical
//! subgraph, and deterministic representative critical path analysis.
//!
//! Weight convention (node-weighted, matching M4 §2.1):
//! - Operation vertex `u` has processing-time weight `p[u] > 0`; dummy `Source` and `Sink`
//!   have zero weight.
//! - `head[u]`: longest processing-time path length from `Source` to `u`, **excluding** `p[u]`.
//! - `tail[u]`: longest processing-time path length from `u` to `Sink`, **excluding** `p[u]`.
//! - `slack[u] = C_max - (head[u] + p[u] + tail[u])`.
//! - Operation `u` is critical iff `slack[u] == 0`.
//! - Directed arc `u -> v` is critical iff it is head-tight (`head[u] + p[u] == head[v]`) and
//!   `v` belongs to a longest path (`head[v] + p[v] + tail[v] == C_max`), equivalently
//!   `head[u] + p[u] + p[v] + tail[v] == C_max` for operation-to-operation arcs.

use crate::error::JsspError;
use crate::graph::critical_blocks::{CriticalBlock, extract_critical_blocks};
use crate::graph::{ArcId, ArcType, DisjunctiveGraph, GraphVertexKind, VertexId};
use crate::model::{JsspInstance, MachineId, OperationId, Schedule};

/// Criticality and maximizing neighbor analysis for one JSSP operation vertex.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperationCriticality {
    /// Canonical operation identifier `(job, op_index)`.
    pub op_id: OperationId,
    /// Graph vertex identifier `VertexId(1 + flat_index)`.
    pub vertex_id: VertexId,
    /// Assigned machine identifier.
    pub machine: MachineId,
    /// Processing time weight `p[u]`.
    pub processing_time: u64,
    /// Longest path from `Source` to `u` excluding `p[u]`.
    pub head: u64,
    /// Longest path from `u` to `Sink` excluding `p[u]`.
    pub tail: u64,
    /// Total schedule slack `C_max - (head[u] + p[u] + tail[u])`.
    pub slack: u64,
    /// True iff `slack == 0`.
    pub is_critical: bool,
    /// All immediate operation predecessors `v` attaining `head[v] + p[v] == head[u]`, sorted
    /// ascending by [`OperationId`].
    pub maximizing_predecessors: Vec<OperationId>,
    /// All incoming [`ArcId`]s attaining `head[u]`, sorted ascending by [`ArcId`].
    pub maximizing_predecessor_arcs: Vec<ArcId>,
    /// Deterministic representative maximizing predecessor (`maximizing_predecessors.first()`).
    pub representative_predecessor: Option<OperationId>,
    /// All immediate operation successors `v` attaining `p[v] + tail[v] == tail[u]`, sorted
    /// ascending by [`OperationId`].
    pub maximizing_successors: Vec<OperationId>,
    /// All outgoing [`ArcId`]s attaining `tail[u]`, sorted ascending by [`ArcId`].
    pub maximizing_successor_arcs: Vec<ArcId>,
    /// Deterministic representative maximizing successor (`maximizing_successors.first()`).
    pub representative_successor: Option<OperationId>,
}

/// Classification result for a directed arc in [`DisjunctiveGraph`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AnalyzedArc {
    /// Arc identifier matching `graph.arcs()[id.0]`.
    pub id: ArcId,
    /// Tail vertex identifier.
    pub from: VertexId,
    /// Head vertex identifier.
    pub to: VertexId,
    /// Tail operation ID (`None` if `from` is `Source`).
    pub from_op: Option<OperationId>,
    /// Head operation ID (`None` if `to` is `Sink`).
    pub to_op: Option<OperationId>,
    /// Semantic constraint class.
    pub arc_type: ArcType,
    /// True iff `head[from] + p[from] == head[to]`.
    pub is_head_tight: bool,
    /// True iff this arc lies on at least one `Source`-to-`Sink` critical path.
    pub is_critical: bool,
}

/// Compact representation of **all** critical `Source`-to-`Sink` paths without path explosion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CriticalSubgraph {
    pub source: VertexId,
    pub sink: VertexId,
    /// All critical operations in canonical [`OperationId`] order.
    pub critical_operations: Vec<OperationId>,
    /// All critical arc IDs (including boundary `Source`/`Sink` arcs) in [`ArcId`] order.
    pub critical_arcs: Vec<ArcId>,
    /// All critical operation-to-operation arcs in [`ArcId`] order.
    pub critical_operation_arcs: Vec<ArcId>,
    /// All critical [`ArcType::JobPrecedence`] arcs in [`ArcId`] order.
    pub critical_job_arcs: Vec<ArcId>,
    /// All critical [`ArcType::MachineOrder`] arcs in [`ArcId`] order.
    pub critical_machine_arcs: Vec<ArcId>,
}

/// One deterministic representative `Source`-to-`Sink` longest path chosen via minimum
/// [`OperationId`] tie-breaking during backward reconstruction from `Sink`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepresentativeCriticalPath {
    /// Operations along the chosen critical path in forward execution order `v_1 .. v_K`.
    pub operations: Vec<OperationId>,
    /// Directed critical arcs `[Source -> v_1, v_1 -> v_2, ..., v_K -> Sink]`.
    pub arcs: Vec<ArcId>,
    /// Sum of operation processing times along `operations` (equals `graph_makespan`).
    pub total_processing_time: u64,
}

/// Complete critical-path, head/tail, critical-subgraph, and critical-block analysis result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CriticalPathAnalysis {
    num_jobs: usize,
    num_machines: usize,
    graph_makespan: u64,
    schedule_makespan: u64,
    topological_order: Vec<VertexId>,
    operations: Vec<OperationCriticality>,
    arcs: Vec<AnalyzedArc>,
    source_maximizing_successors: Vec<OperationId>,
    sink_maximizing_predecessors: Vec<OperationId>,
    critical_subgraph: CriticalSubgraph,
    representative_critical_path: RepresentativeCriticalPath,
    critical_blocks: Vec<CriticalBlock>,
}

impl CriticalPathAnalysis {
    /// Longest `Source`-to-`Sink` path length verified against schedule makespan.
    #[must_use]
    pub const fn graph_makespan(&self) -> u64 {
        self.graph_makespan
    }

    /// Schedule makespan recorded when building the analyzed graph.
    #[must_use]
    pub const fn schedule_makespan(&self) -> u64 {
        self.schedule_makespan
    }

    /// Deterministic topological order of all vertices (`Source`, ops, `Sink`).
    #[must_use]
    pub fn topological_order(&self) -> &[VertexId] {
        &self.topological_order
    }

    /// Criticality records for all `L` operations in canonical `(job, op_index)` order.
    #[must_use]
    pub fn operations(&self) -> &[OperationCriticality] {
        &self.operations
    }

    /// Looks up the criticality record for `op_id`.
    ///
    /// # Errors
    /// Returns [`JsspError`] if `op_id` is out of bounds for the analyzed instance.
    pub fn operation(&self, op_id: OperationId) -> Result<&OperationCriticality, JsspError> {
        let flat = op_id.flat_index(self.num_jobs, self.num_machines)?;
        Ok(&self.operations[flat])
    }

    /// Convenience accessor returning `head[op_id]`.
    ///
    /// # Errors
    /// Returns [`JsspError`] if `op_id` is out of bounds.
    pub fn head(&self, op_id: OperationId) -> Result<u64, JsspError> {
        self.operation(op_id).map(|rec| rec.head)
    }

    /// Convenience accessor returning `tail[op_id]`.
    ///
    /// # Errors
    /// Returns [`JsspError`] if `op_id` is out of bounds.
    pub fn tail(&self, op_id: OperationId) -> Result<u64, JsspError> {
        self.operation(op_id).map(|rec| rec.tail)
    }

    /// Convenience accessor returning `slack[op_id]`.
    ///
    /// # Errors
    /// Returns [`JsspError`] if `op_id` is out of bounds.
    pub fn slack(&self, op_id: OperationId) -> Result<u64, JsspError> {
        self.operation(op_id).map(|rec| rec.slack)
    }

    /// Returns all immediate operation predecessors attaining `head[op_id]`.
    ///
    /// # Errors
    /// Returns [`JsspError`] if `op_id` is out of bounds.
    pub fn maximizing_predecessors(&self, op_id: OperationId) -> Result<&[OperationId], JsspError> {
        self.operation(op_id)
            .map(|rec| rec.maximizing_predecessors.as_slice())
    }

    /// Returns all immediate operation successors attaining `tail[op_id]`.
    ///
    /// # Errors
    /// Returns [`JsspError`] if `op_id` is out of bounds.
    pub fn maximizing_successors(&self, op_id: OperationId) -> Result<&[OperationId], JsspError> {
        self.operation(op_id)
            .map(|rec| rec.maximizing_successors.as_slice())
    }

    /// Operations directly connected to `Source` that initiate a critical path.
    #[must_use]
    pub fn source_maximizing_successors(&self) -> &[OperationId] {
        &self.source_maximizing_successors
    }

    /// Operations directly connected to `Sink` that attain `head[u] + p[u] == C_max`.
    #[must_use]
    pub fn sink_maximizing_predecessors(&self) -> &[OperationId] {
        &self.sink_maximizing_predecessors
    }

    /// All classified arcs in deterministic construction order.
    #[must_use]
    pub fn arcs(&self) -> &[AnalyzedArc] {
        &self.arcs
    }

    /// Compact critical subgraph containing all critical operations and critical arcs.
    #[must_use]
    pub const fn critical_subgraph(&self) -> &CriticalSubgraph {
        &self.critical_subgraph
    }

    /// All critical operations (`slack == 0`) in canonical [`OperationId`] order.
    #[must_use]
    pub fn critical_operations(&self) -> &[OperationId] {
        &self.critical_subgraph.critical_operations
    }

    /// All critical arcs (`Source`, `JobPrecedence`, `MachineOrder`, `Sink`).
    #[must_use]
    pub fn critical_arcs(&self) -> Vec<AnalyzedArc> {
        self.critical_subgraph
            .critical_arcs
            .iter()
            .map(|&id| self.arcs[id.0])
            .collect()
    }

    /// All critical [`ArcType::MachineOrder`] arcs in [`ArcId`] order.
    #[must_use]
    pub fn critical_machine_arcs(&self) -> Vec<AnalyzedArc> {
        self.critical_subgraph
            .critical_machine_arcs
            .iter()
            .map(|&id| self.arcs[id.0])
            .collect()
    }

    /// All critical [`ArcType::JobPrecedence`] arcs in [`ArcId`] order.
    #[must_use]
    pub fn critical_job_arcs(&self) -> Vec<AnalyzedArc> {
        self.critical_subgraph
            .critical_job_arcs
            .iter()
            .map(|&id| self.arcs[id.0])
            .collect()
    }

    /// Deterministic representative longest path from `Source` to `Sink`.
    #[must_use]
    pub const fn representative_critical_path(&self) -> &RepresentativeCriticalPath {
        &self.representative_critical_path
    }

    /// Maximal contiguous critical blocks (`len >= 2`) extracted across all machines.
    #[must_use]
    pub fn critical_blocks(&self) -> &[CriticalBlock] {
        &self.critical_blocks
    }
}

/// Validates a schedule, constructs its compact oriented disjunctive graph, and performs full
/// critical-path, head/tail, critical-subgraph, and critical-block analysis.
///
/// # Errors
/// Returns [`JsspError`] if schedule validation, DAG verification, makespan equivalence, or
/// checked `u64` arithmetic fails.
pub fn analyze_schedule(
    instance: &JsspInstance,
    schedule: &Schedule,
) -> Result<(DisjunctiveGraph, CriticalPathAnalysis), JsspError> {
    let graph = DisjunctiveGraph::from_schedule(instance, schedule)?;
    let analysis = analyze_disjunctive_graph(&graph)?;
    Ok((graph, analysis))
}

/// Computes topological order, head/tail times, slack, critical operations/arcs, compact
/// critical subgraph, representative critical path, and critical blocks on `graph`.
///
/// Complexity: `O((V + E) log V)` for topological sorting plus `O(V + E)` for head/tail DP,
/// arc classification, and critical-block extraction (since every vertex has in-degree and
/// out-degree `<= 2` except dummy `Source` and `Sink`).
///
/// # Errors
/// Returns [`JsspError`] on directed cycles, `u64` arithmetic overflow, makespan mismatch
/// (`Cmax_graph != Cmax_schedule`), or negative operation slack.
pub fn analyze_disjunctive_graph(
    graph: &DisjunctiveGraph,
) -> Result<CriticalPathAnalysis, JsspError> {
    let meta = *graph.metadata();
    if meta.num_operations == 0 {
        return Err(JsspError::EmptySchedule);
    }

    let topo_order = graph.topological_sort()?;
    let num_vertices = graph.vertices().len();
    let source_id = graph.source_vertex_id();
    let sink_id = graph.sink_vertex_id();

    let (heads, max_pred_arcs, max_pred_ops) = compute_heads(graph, &topo_order, num_vertices)?;
    let (tails, max_succ_arcs, max_succ_ops) = compute_tails(graph, &topo_order, num_vertices)?;

    let cmax_sink = heads[sink_id.0];
    let cmax_source = tails[source_id.0];
    if cmax_sink != cmax_source {
        return Err(JsspError::InvalidDisjunctiveGraph {
            detail: format!(
                "forward longest path ({cmax_sink}) disagrees with reverse longest path ({cmax_source})"
            ),
        });
    }

    let mut cmax_ops: u64 = 0;
    for vertex in graph.vertices() {
        if let GraphVertexKind::Operation(_) = vertex.kind {
            let total = heads[vertex.id.0]
                .checked_add(vertex.processing_time)
                .and_then(|x| x.checked_add(tails[vertex.id.0]))
                .ok_or(JsspError::TimeOverflow)?;
            if total > cmax_ops {
                cmax_ops = total;
            }
        }
    }

    if cmax_ops != cmax_sink {
        return Err(JsspError::InvalidDisjunctiveGraph {
            detail: format!(
                "max operation path length ({cmax_ops}) disagrees with Source-to-Sink path length ({cmax_sink})"
            ),
        });
    }

    let cmax_graph = cmax_sink;
    if cmax_graph != meta.schedule_makespan {
        return Err(JsspError::GraphMakespanMismatch {
            schedule_makespan: meta.schedule_makespan,
            graph_makespan: cmax_graph,
        });
    }

    let operations = build_operation_criticality(
        graph,
        &heads,
        &tails,
        &max_pred_ops,
        &max_pred_arcs,
        &max_succ_ops,
        &max_succ_arcs,
        cmax_graph,
    )?;

    let analyzed_arcs = classify_arcs(graph, &heads, &tails, cmax_graph)?;
    let critical_subgraph = build_critical_subgraph(graph, &operations, &analyzed_arcs);
    let representative_critical_path = build_representative_critical_path(
        graph,
        &operations,
        &analyzed_arcs,
        &max_pred_ops[sink_id.0],
        cmax_graph,
    )?;
    let critical_blocks = extract_critical_blocks(graph, &analyzed_arcs);

    Ok(CriticalPathAnalysis {
        num_jobs: meta.num_jobs,
        num_machines: meta.num_machines,
        graph_makespan: cmax_graph,
        schedule_makespan: meta.schedule_makespan,
        topological_order: topo_order,
        operations,
        arcs: analyzed_arcs,
        source_maximizing_successors: max_succ_ops[source_id.0].clone(),
        sink_maximizing_predecessors: max_pred_ops[sink_id.0].clone(),
        critical_subgraph,
        representative_critical_path,
        critical_blocks,
    })
}

type DPPredecessorPass = (Vec<u64>, Vec<Vec<ArcId>>, Vec<Vec<OperationId>>);

fn compute_heads(
    graph: &DisjunctiveGraph,
    topo_order: &[VertexId],
    num_vertices: usize,
) -> Result<DPPredecessorPass, JsspError> {
    let mut heads = vec![0u64; num_vertices];
    let mut max_arcs = vec![Vec::new(); num_vertices];
    let mut max_ops = vec![Vec::new(); num_vertices];

    for &u in topo_order {
        let incoming = graph.incoming_arcs(u)?;
        if incoming.is_empty() {
            heads[u.0] = 0;
            continue;
        }

        let mut best = 0u64;
        let mut candidates: Vec<(ArcId, Option<OperationId>, u64)> =
            Vec::with_capacity(incoming.len());

        for &arc_id in incoming {
            let arc = graph.arc(arc_id)?;
            let pred_v = graph.vertex(arc.from)?;
            let cand = heads[pred_v.id.0]
                .checked_add(pred_v.processing_time)
                .ok_or(JsspError::TimeOverflow)?;
            if cand > best {
                best = cand;
            }
            candidates.push((arc_id, arc.from_op, cand));
        }

        heads[u.0] = best;
        for (arc_id, from_op, cand) in candidates {
            if cand == best {
                max_arcs[u.0].push(arc_id);
                if let Some(op_id) = from_op {
                    max_ops[u.0].push(op_id);
                }
            }
        }

        max_arcs[u.0].sort();
        max_ops[u.0].sort();
        max_ops[u.0].dedup();
    }

    Ok((heads, max_arcs, max_ops))
}

fn compute_tails(
    graph: &DisjunctiveGraph,
    topo_order: &[VertexId],
    num_vertices: usize,
) -> Result<DPPredecessorPass, JsspError> {
    let mut tails = vec![0u64; num_vertices];
    let mut max_arcs = vec![Vec::new(); num_vertices];
    let mut max_ops = vec![Vec::new(); num_vertices];

    for &u in topo_order.iter().rev() {
        let outgoing = graph.outgoing_arcs(u)?;
        if outgoing.is_empty() {
            tails[u.0] = 0;
            continue;
        }

        let mut best = 0u64;
        let mut candidates: Vec<(ArcId, Option<OperationId>, u64)> =
            Vec::with_capacity(outgoing.len());

        for &arc_id in outgoing {
            let arc = graph.arc(arc_id)?;
            let succ_v = graph.vertex(arc.to)?;
            let cand = succ_v
                .processing_time
                .checked_add(tails[succ_v.id.0])
                .ok_or(JsspError::TimeOverflow)?;
            if cand > best {
                best = cand;
            }
            candidates.push((arc_id, arc.to_op, cand));
        }

        tails[u.0] = best;
        for (arc_id, to_op, cand) in candidates {
            if cand == best {
                max_arcs[u.0].push(arc_id);
                if let Some(op_id) = to_op {
                    max_ops[u.0].push(op_id);
                }
            }
        }

        max_arcs[u.0].sort();
        max_ops[u.0].sort();
        max_ops[u.0].dedup();
    }

    Ok((tails, max_arcs, max_ops))
}

#[allow(clippy::too_many_arguments)]
fn build_operation_criticality(
    graph: &DisjunctiveGraph,
    heads: &[u64],
    tails: &[u64],
    max_pred_ops: &[Vec<OperationId>],
    max_pred_arcs: &[Vec<ArcId>],
    max_succ_ops: &[Vec<OperationId>],
    max_succ_arcs: &[Vec<ArcId>],
    cmax_graph: u64,
) -> Result<Vec<OperationCriticality>, JsspError> {
    let meta = graph.metadata();
    let mut operations = Vec::with_capacity(meta.num_operations);

    for flat in 0..meta.num_operations {
        let v_id = VertexId(flat + 1);
        let vertex = graph.vertex(v_id)?;
        let op_id = vertex
            .operation_id()
            .ok_or_else(|| JsspError::InvalidDisjunctiveGraph {
                detail: format!("expected operation vertex at index {}", v_id.0),
            })?;
        let machine = vertex
            .machine
            .ok_or_else(|| JsspError::InvalidDisjunctiveGraph {
                detail: format!("operation vertex {} missing machine metadata", v_id.0),
            })?;

        let head = heads[v_id.0];
        let tail = tails[v_id.0];
        let path_len = head
            .checked_add(vertex.processing_time)
            .and_then(|x| x.checked_add(tail))
            .ok_or(JsspError::TimeOverflow)?;
        if path_len > cmax_graph {
            return Err(JsspError::NegativeOperationSlack {
                job: op_id.job.0,
                op_index: op_id.op_index.0,
                head_plus_proc_plus_tail: path_len,
                graph_makespan: cmax_graph,
            });
        }

        let slack = cmax_graph - path_len;
        let preds = max_pred_ops[v_id.0].clone();
        let succs = max_succ_ops[v_id.0].clone();

        operations.push(OperationCriticality {
            op_id,
            vertex_id: v_id,
            machine,
            processing_time: vertex.processing_time,
            head,
            tail,
            slack,
            is_critical: slack == 0,
            representative_predecessor: preds.first().copied(),
            maximizing_predecessors: preds,
            maximizing_predecessor_arcs: max_pred_arcs[v_id.0].clone(),
            representative_successor: succs.first().copied(),
            maximizing_successors: succs,
            maximizing_successor_arcs: max_succ_arcs[v_id.0].clone(),
        });
    }

    Ok(operations)
}

fn classify_arcs(
    graph: &DisjunctiveGraph,
    heads: &[u64],
    tails: &[u64],
    cmax_graph: u64,
) -> Result<Vec<AnalyzedArc>, JsspError> {
    let mut analyzed = Vec::with_capacity(graph.arcs().len());

    for &arc in graph.arcs() {
        let from_v = graph.vertex(arc.from)?;
        let to_v = graph.vertex(arc.to)?;

        let head_plus_p = heads[from_v.id.0]
            .checked_add(from_v.processing_time)
            .ok_or(JsspError::TimeOverflow)?;
        let is_head_tight = head_plus_p == heads[to_v.id.0];

        let is_critical = match arc.arc_type {
            ArcType::JobPrecedence { .. } | ArcType::MachineOrder { .. } => {
                let to_total = heads[to_v.id.0]
                    .checked_add(to_v.processing_time)
                    .and_then(|x| x.checked_add(tails[to_v.id.0]))
                    .ok_or(JsspError::TimeOverflow)?;
                if to_total > cmax_graph {
                    let op_id = arc.to_op.unwrap_or(OperationId::from_indices(0, 0));
                    return Err(JsspError::NegativeOperationSlack {
                        job: op_id.job.0,
                        op_index: op_id.op_index.0,
                        head_plus_proc_plus_tail: to_total,
                        graph_makespan: cmax_graph,
                    });
                }
                let arc_total = head_plus_p
                    .checked_add(to_v.processing_time)
                    .and_then(|x| x.checked_add(tails[to_v.id.0]))
                    .ok_or(JsspError::TimeOverflow)?;
                if arc_total > cmax_graph {
                    return Err(JsspError::InvalidDisjunctiveGraph {
                        detail: format!(
                            "arc {} path weight {arc_total} exceeds graph makespan {cmax_graph}",
                            arc.id.0
                        ),
                    });
                }
                let critical_via_head_and_v = is_head_tight && (to_total == cmax_graph);
                let critical_via_sum = arc_total == cmax_graph;
                if critical_via_head_and_v != critical_via_sum {
                    return Err(JsspError::InvalidDisjunctiveGraph {
                        detail: format!(
                            "arc {} criticality inconsistency between head-tightness ({critical_via_head_and_v}) and arc sum ({critical_via_sum})",
                            arc.id.0
                        ),
                    });
                }
                critical_via_head_and_v
            }
            ArcType::Source => {
                let to_total = heads[to_v.id.0]
                    .checked_add(to_v.processing_time)
                    .and_then(|x| x.checked_add(tails[to_v.id.0]))
                    .ok_or(JsspError::TimeOverflow)?;
                is_head_tight && (to_total == cmax_graph)
            }
            ArcType::Sink => {
                let from_total = head_plus_p
                    .checked_add(tails[from_v.id.0])
                    .ok_or(JsspError::TimeOverflow)?;
                is_head_tight && (tails[from_v.id.0] == 0) && (from_total == cmax_graph)
            }
        };

        analyzed.push(AnalyzedArc {
            id: arc.id,
            from: arc.from,
            to: arc.to,
            from_op: arc.from_op,
            to_op: arc.to_op,
            arc_type: arc.arc_type,
            is_head_tight,
            is_critical,
        });
    }

    Ok(analyzed)
}

fn build_critical_subgraph(
    graph: &DisjunctiveGraph,
    operations: &[OperationCriticality],
    analyzed_arcs: &[AnalyzedArc],
) -> CriticalSubgraph {
    let critical_operations = operations
        .iter()
        .filter(|op| op.is_critical)
        .map(|op| op.op_id)
        .collect();

    let mut critical_arcs = Vec::new();
    let mut critical_operation_arcs = Vec::new();
    let mut critical_job_arcs = Vec::new();
    let mut critical_machine_arcs = Vec::new();

    for &arc in analyzed_arcs {
        if !arc.is_critical {
            continue;
        }
        critical_arcs.push(arc.id);
        match arc.arc_type {
            ArcType::JobPrecedence { .. } => {
                critical_operation_arcs.push(arc.id);
                critical_job_arcs.push(arc.id);
            }
            ArcType::MachineOrder { .. } => {
                critical_operation_arcs.push(arc.id);
                critical_machine_arcs.push(arc.id);
            }
            ArcType::Source | ArcType::Sink => {}
        }
    }

    CriticalSubgraph {
        source: graph.source_vertex_id(),
        sink: graph.sink_vertex_id(),
        critical_operations,
        critical_arcs,
        critical_operation_arcs,
        critical_job_arcs,
        critical_machine_arcs,
    }
}

fn build_representative_critical_path(
    graph: &DisjunctiveGraph,
    operations: &[OperationCriticality],
    analyzed_arcs: &[AnalyzedArc],
    sink_max_preds: &[OperationId],
    cmax_graph: u64,
) -> Result<RepresentativeCriticalPath, JsspError> {
    let meta = graph.metadata();
    let sink_id = graph.sink_vertex_id();
    let mut current_op =
        *sink_max_preds
            .first()
            .ok_or_else(|| JsspError::InvalidDisjunctiveGraph {
                detail: "sink has no maximizing operation predecessor".to_owned(),
            })?;

    let find_critical_arc = |from: VertexId, to: VertexId| -> Result<ArcId, JsspError> {
        graph
            .outgoing_arcs(from)?
            .iter()
            .copied()
            .find(|&arc_id| {
                let arc = analyzed_arcs[arc_id.0];
                arc.to == to && arc.is_critical
            })
            .ok_or_else(|| JsspError::InvalidDisjunctiveGraph {
                detail: format!(
                    "missing critical arc from vertex {} to vertex {}",
                    from.0, to.0
                ),
            })
    };

    let mut ops_rev = Vec::new();
    let mut arcs_rev = Vec::new();

    // Terminal Sink arc from final critical operation v_K -> Sink.
    let v_last = VertexId::from_operation(current_op, meta.num_jobs, meta.num_machines)?;
    arcs_rev.push(find_critical_arc(v_last, sink_id)?);

    let mut guard = 0usize;
    loop {
        guard += 1;
        if guard > meta.num_operations {
            return Err(JsspError::InvalidDisjunctiveGraph {
                detail: "cycle encountered while reconstructing representative critical path"
                    .to_owned(),
            });
        }

        ops_rev.push(current_op);
        let flat = current_op.flat_index(meta.num_jobs, meta.num_machines)?;
        let rec = &operations[flat];

        if let Some(prev_op) = rec.representative_predecessor {
            let from_v = VertexId::from_operation(prev_op, meta.num_jobs, meta.num_machines)?;
            arcs_rev.push(find_critical_arc(from_v, rec.vertex_id)?);
            current_op = prev_op;
        } else {
            arcs_rev.push(find_critical_arc(graph.source_vertex_id(), rec.vertex_id)?);
            break;
        }
    }

    ops_rev.reverse();
    arcs_rev.reverse();

    let mut total_processing_time = 0u64;
    for &op_id in &ops_rev {
        let flat = op_id.flat_index(meta.num_jobs, meta.num_machines)?;
        total_processing_time = total_processing_time
            .checked_add(operations[flat].processing_time)
            .ok_or(JsspError::TimeOverflow)?;
    }

    if total_processing_time != cmax_graph {
        return Err(JsspError::InvalidDisjunctiveGraph {
            detail: format!(
                "representative critical path processing sum ({total_processing_time}) != graph makespan ({cmax_graph})"
            ),
        });
    }

    Ok(RepresentativeCriticalPath {
        operations: ops_rev,
        arcs: arcs_rev,
        total_processing_time,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{JsspInstance, MachineId, MachineSequence};

    #[test]
    fn checked_overflow_and_negative_slack_helpers_reject_inconsistent_graphs() {
        let instance =
            JsspInstance::new(vec![vec![(0, 10), (1, 1)], vec![(0, 1), (1, 1)]]).expect("2x2");
        let seqs = vec![
            MachineSequence::new(
                MachineId(0),
                vec![
                    OperationId::from_indices(0, 0),
                    OperationId::from_indices(1, 0),
                ],
            ),
            MachineSequence::new(
                MachineId(1),
                vec![
                    OperationId::from_indices(1, 1),
                    OperationId::from_indices(0, 1),
                ],
            ),
        ];

        // Graph makespan mismatch error test when schedule_makespan is inconsistent.
        let mismatched =
            DisjunctiveGraph::from_machine_sequences(&instance, &seqs, 999).expect("dag");
        assert_eq!(
            analyze_disjunctive_graph(&mismatched),
            Err(JsspError::GraphMakespanMismatch {
                schedule_makespan: 999,
                graph_makespan: 13,
            })
        );

        // White-box overflow verification in DP: mutate synthetic graph vertex weights.
        let mut overflowing = mismatched.clone();
        overflowing.vertices[1].processing_time = u64::MAX;
        overflowing.vertices[2].processing_time = 1;
        let topo = overflowing.topological_sort().expect("acyclic");
        assert_eq!(
            compute_heads(&overflowing, &topo, 6),
            Err(JsspError::TimeOverflow)
        );
        assert_eq!(
            compute_tails(&overflowing, &topo, 6),
            Err(JsspError::TimeOverflow)
        );
        assert_eq!(
            analyze_disjunctive_graph(&overflowing),
            Err(JsspError::TimeOverflow)
        );

        // White-box negative slack guard verification in build_operation_criticality.
        let heads = vec![0, 5, 0, 0, 0, 13];
        let tails = vec![13, 5, 0, 0, 0, 0];
        let empty_ops = vec![Vec::new(); 6];
        let empty_arcs = vec![Vec::new(); 6];
        let err = build_operation_criticality(
            &mismatched,
            &heads,
            &tails,
            &empty_ops,
            &empty_arcs,
            &empty_ops,
            &empty_arcs,
            13,
        )
        .expect_err("head(5) + p(10) + tail(5) = 20 > 13 must trigger NegativeOperationSlack");
        assert_eq!(
            err,
            JsspError::NegativeOperationSlack {
                job: 0,
                op_index: 0,
                head_plus_proc_plus_tail: 20,
                graph_makespan: 13,
            }
        );
    }
}
