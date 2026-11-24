//! Compact oriented disjunctive graph representation for deterministic classical JSSP.
//!
//! Given a validated schedule `S` for an instance `I` with `J` jobs and `M` machines
//! (`L = J * M` operations), [`DisjunctiveGraph`] constructs the oriented scheduling DAG
//! containing:
//! - One dummy `Source` vertex (`VertexId(0)`, weight `0`),
//! - One operation vertex per `O(j, k)` (`VertexId(1 + flat_index)`, weight `p[u] > 0`),
//! - One dummy `Sink` vertex (`VertexId(L + 1)`, weight `0`),
//! - All intra-job conjunctive [`ArcType::JobPrecedence`] arcs `O(j, k) -> O(j, k + 1)`,
//! - Directed [`ArcType::MachineOrder`] arcs between consecutive operations in each
//!   chronological machine sequence from `S`, and
//! - [`ArcType::Source`] / [`ArcType::Sink`] boundary arcs without duplicate edges.
//!
//! Storage uses deterministic adjacency lists (`O(V + E)` memory and `O(V + E)` DP
//! traversals; `O((V + E) log V)` topological sorting due to deterministic `OperationId`
//! ready-queue tie-breaking).

pub mod critical_blocks;
pub mod critical_path;

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::fmt;

pub use critical_blocks::{CriticalBlock, CriticalBlockId, extract_critical_blocks};
pub use critical_path::{
    AnalyzedArc, CriticalPathAnalysis, CriticalSubgraph, OperationCriticality,
    RepresentativeCriticalPath, analyze_disjunctive_graph, analyze_schedule,
};

use crate::error::JsspError;
use crate::feasibility::validate_schedule;
use crate::model::{JobId, JsspInstance, MachineId, MachineSequence, OperationId, Schedule};

/// Strongly typed 0-based identifier for a vertex in a [`DisjunctiveGraph`].
///
/// Layout is deterministic and order-isomorphic to [`OperationId`]:
/// - `VertexId(0)`: dummy `Source` vertex
/// - `VertexId(1 + flat_index)`: operation `OperationId::from_flat_index(flat_index, J, M)`
/// - `VertexId(L + 1)`: dummy `Sink` vertex
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VertexId(pub usize);

impl VertexId {
    /// Canonical identifier of the dummy `Source` vertex (`VertexId(0)`).
    pub const SOURCE: Self = Self(0);

    /// Constructs a `VertexId` from a raw vertex index.
    #[must_use]
    pub const fn new(index: usize) -> Self {
        Self(index)
    }

    /// Returns the underlying zero-based vertex index.
    #[must_use]
    pub const fn index(self) -> usize {
        self.0
    }

    /// Returns the canonical `Sink` vertex identifier `VertexId(num_operations + 1)`.
    ///
    /// # Errors
    /// Returns [`JsspError::DimensionOverflow`] if `num_operations + 1` overflows `usize`.
    pub fn sink(num_operations: usize) -> Result<Self, JsspError> {
        num_operations
            .checked_add(1)
            .map(Self)
            .ok_or(JsspError::DimensionOverflow)
    }

    /// Maps canonical operation identifier `op_id` to `VertexId(1 + flat_index)`.
    ///
    /// # Errors
    /// Returns [`JsspError`] if `op_id` is out of bounds or dimensions overflow.
    pub fn from_operation(
        op_id: OperationId,
        num_jobs: usize,
        num_machines: usize,
    ) -> Result<Self, JsspError> {
        let flat = op_id.flat_index(num_jobs, num_machines)?;
        flat.checked_add(1)
            .map(Self)
            .ok_or(JsspError::DimensionOverflow)
    }
}

impl From<usize> for VertexId {
    fn from(value: usize) -> Self {
        Self(value)
    }
}

impl fmt::Display for VertexId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "V{}", self.0)
    }
}

/// Strongly typed 0-based identifier for a directed arc in a [`DisjunctiveGraph`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ArcId(pub usize);

impl ArcId {
    /// Constructs an `ArcId` from a raw zero-based index.
    #[must_use]
    pub const fn new(index: usize) -> Self {
        Self(index)
    }

    /// Returns the underlying zero-based arc index.
    #[must_use]
    pub const fn index(self) -> usize {
        self.0
    }
}

impl From<usize> for ArcId {
    fn from(value: usize) -> Self {
        Self(value)
    }
}

impl fmt::Display for ArcId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "A{}", self.0)
    }
}

/// Classification of a vertex in a [`DisjunctiveGraph`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GraphVertexKind {
    /// Dummy start vertex with zero processing-time weight.
    Source,
    /// Scheduled JSSP operation with strictly positive processing-time weight.
    Operation(OperationId),
    /// Dummy terminal vertex with zero processing-time weight.
    Sink,
}

/// Metadata attached to a single vertex in a [`DisjunctiveGraph`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GraphVertex {
    /// Unique vertex identifier.
    pub id: VertexId,
    /// Role of this vertex (`Source`, `Operation`, or `Sink`).
    pub kind: GraphVertexKind,
    /// Assigned machine for operation vertices; `None` for dummy vertices.
    pub machine: Option<MachineId>,
    /// Processing-time node weight `p[u]` (`0` for dummy `Source` / `Sink`).
    pub processing_time: u64,
}

impl GraphVertex {
    /// Returns the canonical [`OperationId`] if this is an operation vertex.
    #[must_use]
    pub const fn operation_id(&self) -> Option<OperationId> {
        match self.kind {
            GraphVertexKind::Operation(op_id) => Some(op_id),
            GraphVertexKind::Source | GraphVertexKind::Sink => None,
        }
    }
}

/// Semantic origin of a directed arc in a compact oriented disjunctive graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ArcType {
    /// Boundary arc from `Source` to an operation without incoming job or machine arcs.
    Source,
    /// Conjunctive intra-job precedence constraint `O(j, k) -> O(j, k + 1)`.
    JobPrecedence {
        /// Owning job identifier.
        job: JobId,
    },
    /// Oriented disjunctive arc between consecutive operations on one machine sequence.
    MachineOrder {
        /// Machine identifier carrying this precedence relation.
        machine: MachineId,
    },
    /// Boundary arc from an operation without outgoing job or machine arcs to `Sink`.
    Sink,
}

impl ArcType {
    /// Returns a stable canonical string identifier for serialization boundaries.
    #[must_use]
    pub const fn kind_str(&self) -> &'static str {
        match self {
            Self::Source => "source",
            Self::JobPrecedence { .. } => "job_precedence",
            Self::MachineOrder { .. } => "machine_order",
            Self::Sink => "sink",
        }
    }

    /// Returns the numeric owner identifier (`job` or `machine`), or `None` for boundary arcs.
    #[must_use]
    pub const fn owner_index(&self) -> Option<usize> {
        match self {
            Self::JobPrecedence { job } => Some(job.0),
            Self::MachineOrder { machine } => Some(machine.0),
            Self::Source | Self::Sink => None,
        }
    }
}

/// Directed arc inside a [`DisjunctiveGraph`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GraphArc {
    /// Unique arc identifier.
    pub id: ArcId,
    /// Tail vertex identifier.
    pub from: VertexId,
    /// Head vertex identifier.
    pub to: VertexId,
    /// Canonical [`OperationId`] of `from`, or `None` when `from` is `Source`.
    pub from_op: Option<OperationId>,
    /// Canonical [`OperationId`] of `to`, or `None` when `to` is `Sink`.
    pub to_op: Option<OperationId>,
    /// Semantic constraint class (`Source`, `JobPrecedence`, `MachineOrder`, or `Sink`).
    pub arc_type: ArcType,
}

/// Summary counts and schedule makespan attached to a [`DisjunctiveGraph`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GraphMetadata {
    pub num_jobs: usize,
    pub num_machines: usize,
    pub num_operations: usize,
    pub num_vertices: usize,
    pub num_arcs: usize,
    pub num_job_precedence_arcs: usize,
    pub num_machine_order_arcs: usize,
    pub num_source_arcs: usize,
    pub num_sink_arcs: usize,
    pub schedule_makespan: u64,
}

/// Compact oriented disjunctive graph using deterministic `O(V + E)` adjacency lists.
///
/// The graph stores conjunctive intra-job precedence arcs, consecutive directed machine-order
/// arcs derived from the schedule's chronological machine sequences, and zero-weight dummy
/// `Source` / `Sink` boundary arcs. It does not explicitly materialize all `O(J^2)` transitive
/// machine pairs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DisjunctiveGraph {
    metadata: GraphMetadata,
    vertices: Vec<GraphVertex>,
    arcs: Vec<GraphArc>,
    outgoing: Vec<Vec<ArcId>>,
    incoming: Vec<Vec<ArcId>>,
    machine_sequences: Vec<MachineSequence>,
    machine_order_arcs: Vec<Vec<ArcId>>,
}

impl DisjunctiveGraph {
    /// Builds and verifies an acyclic compact oriented disjunctive graph from a schedule.
    ///
    /// Validation delegates exclusively to [`validate_schedule`] before extraction, and the
    /// resulting graph is verified acyclic via [`DisjunctiveGraph::topological_sort`].
    ///
    /// # Errors
    /// Returns [`JsspError`] if `schedule` fails M2 validation, dimensions overflow, or a
    /// directed cycle is detected.
    pub fn from_schedule(instance: &JsspInstance, schedule: &Schedule) -> Result<Self, JsspError> {
        validate_schedule(instance, schedule)?;
        let machine_sequences = schedule.machine_sequences();
        let graph =
            Self::build_from_machine_sequences(instance, &machine_sequences, schedule.makespan())?;
        let _ = graph.topological_sort()?;
        Ok(graph)
    }

    /// Builds a validated acyclic disjunctive graph directly from per-machine sequences.
    ///
    /// # Errors
    /// Returns [`JsspError`] if machine sequences are invalid or induce a directed cycle.
    pub fn from_machine_sequences(
        instance: &JsspInstance,
        machine_sequences: &[MachineSequence],
        schedule_makespan: u64,
    ) -> Result<Self, JsspError> {
        let graph =
            Self::build_from_machine_sequences(instance, machine_sequences, schedule_makespan)?;
        let _ = graph.topological_sort()?;
        Ok(graph)
    }

    /// Builds and validates a target acyclic disjunctive graph from complete per-machine
    /// operation sequences, computing and attaching the exact longest-path makespan.
    ///
    /// # Errors
    /// Returns [`JsspError::InvalidDisjunctiveGraph`], [`JsspError::UnexpectedOperation`],
    /// [`JsspError::OperationMachineMismatch`], [`JsspError::DuplicateOperation`],
    /// [`JsspError::MissingOperation`], [`JsspError::GraphCycleDetected`], or
    /// [`JsspError::TimeOverflow`].
    pub fn from_target_machine_sequences(
        instance: &JsspInstance,
        machine_sequences: &[MachineSequence],
    ) -> Result<Self, JsspError> {
        let mut graph = Self::build_from_machine_sequences(instance, machine_sequences, 0)?;
        let topo = graph.topological_sort()?;
        let num_vertices = graph.vertices.len();
        let mut heads = vec![0u64; num_vertices];
        for &u in &topo {
            let incoming = graph.incoming_arcs(u)?;
            if incoming.is_empty() {
                continue;
            }
            let mut best = 0u64;
            for &arc_id in incoming {
                let pred = graph.arcs[arc_id.0].from;
                let cand = heads[pred.0]
                    .checked_add(graph.vertices[pred.0].processing_time)
                    .ok_or(JsspError::TimeOverflow)?;
                if cand > best {
                    best = cand;
                }
            }
            heads[u.0] = best;
        }
        graph.metadata.schedule_makespan = heads[graph.sink_vertex_id().0];
        Ok(graph)
    }

    /// Builds a compact oriented disjunctive graph from machine sequences **without**
    /// running acyclicity verification at construction time.
    ///
    /// Intended for cycle-detection verification and structural diagnostics where callers
    /// deliberately construct malformed cyclic orientations and inspect [`DisjunctiveGraph::topological_sort`].
    ///
    /// # Errors
    /// Returns [`JsspError::InvalidDisjunctiveGraph`] or [`JsspError::DimensionOverflow`] if
    /// machine sequences violate coverage or machine-assignment invariants.
    pub fn from_machine_sequences_unchecked(
        instance: &JsspInstance,
        machine_sequences: &[MachineSequence],
        schedule_makespan: u64,
    ) -> Result<Self, JsspError> {
        Self::build_from_machine_sequences(instance, machine_sequences, schedule_makespan)
    }

    fn build_from_machine_sequences(
        instance: &JsspInstance,
        machine_sequences: &[MachineSequence],
        schedule_makespan: u64,
    ) -> Result<Self, JsspError> {
        let num_jobs = instance.num_jobs();
        let num_machines = instance.num_machines();
        let num_operations = instance.num_operations();

        if machine_sequences.len() != num_machines {
            return Err(JsspError::InvalidDisjunctiveGraph {
                detail: format!(
                    "expected {num_machines} machine sequences, got {}",
                    machine_sequences.len()
                ),
            });
        }

        let sink_vertex = VertexId::sink(num_operations)?;
        let num_vertices = sink_vertex
            .0
            .checked_add(1)
            .ok_or(JsspError::DimensionOverflow)?;

        let mut vertices = Vec::with_capacity(num_vertices);
        vertices.push(GraphVertex {
            id: VertexId::SOURCE,
            kind: GraphVertexKind::Source,
            machine: None,
            processing_time: 0,
        });

        for flat in 0..num_operations {
            let op_id = OperationId::from_flat_index(flat, num_jobs, num_machines)?;
            let op_spec = instance.operation_by_indices(op_id.job.0, op_id.op_index.0)?;
            vertices.push(GraphVertex {
                id: VertexId(flat + 1),
                kind: GraphVertexKind::Operation(op_id),
                machine: Some(op_spec.machine),
                processing_time: op_spec.processing_time,
            });
        }

        vertices.push(GraphVertex {
            id: sink_vertex,
            kind: GraphVertexKind::Sink,
            machine: None,
            processing_time: 0,
        });

        // Validate machine sequence coverage and machine ownership.
        let mut seen = vec![false; num_operations];
        let mut normalized_sequences = Vec::with_capacity(num_machines);
        for (m, seq) in machine_sequences.iter().enumerate() {
            if seq.machine.0 != m {
                return Err(JsspError::InvalidDisjunctiveGraph {
                    detail: format!(
                        "machine sequence at index {m} reports machine {}, expected {m}",
                        seq.machine.0
                    ),
                });
            }
            if seq.operations.len() != num_jobs {
                return Err(JsspError::InvalidDisjunctiveGraph {
                    detail: format!(
                        "machine {m} sequence has {} operations, expected {num_jobs}",
                        seq.operations.len()
                    ),
                });
            }
            for &op_id in &seq.operations {
                let op_spec = instance
                    .operation(op_id)
                    .ok_or(JsspError::UnexpectedOperation {
                        job: op_id.job.0,
                        op_index: op_id.op_index.0,
                        num_jobs,
                        num_machines,
                    })?;
                if op_spec.machine.0 != m {
                    return Err(JsspError::OperationMachineMismatch {
                        job: op_id.job.0,
                        op_index: op_id.op_index.0,
                        expected_machine: op_spec.machine.0,
                        actual_machine: m,
                    });
                }
                let flat = op_id.flat_index(num_jobs, num_machines)?;
                if seen[flat] {
                    return Err(JsspError::DuplicateOperation {
                        job: op_id.job.0,
                        op_index: op_id.op_index.0,
                    });
                }
                seen[flat] = true;
            }
            normalized_sequences.push(seq.clone());
        }

        if let Some((missing_flat, _)) = seen.iter().enumerate().find(|&(_, &present)| !present) {
            let missing_op = OperationId::from_flat_index(missing_flat, num_jobs, num_machines)?;
            return Err(JsspError::MissingOperation {
                job: missing_op.job.0,
                op_index: missing_op.op_index.0,
            });
        }

        let mut arcs = Vec::new();
        let mut outgoing = vec![Vec::new(); num_vertices];
        let mut incoming = vec![Vec::new(); num_vertices];
        let mut machine_order_arcs = vec![Vec::new(); num_machines];

        let mut num_job_precedence_arcs = 0usize;
        let mut num_machine_order_arcs = 0usize;
        let mut num_source_arcs = 0usize;
        let mut num_sink_arcs = 0usize;

        // Helper closure avoiding duplicate (from, to, arc_type) arcs.
        let push_arc = |arcs: &mut Vec<GraphArc>,
                        outgoing: &mut Vec<Vec<ArcId>>,
                        incoming: &mut Vec<Vec<ArcId>>,
                        from: VertexId,
                        to: VertexId,
                        from_op: Option<OperationId>,
                        to_op: Option<OperationId>,
                        arc_type: ArcType|
         -> Option<ArcId> {
            let duplicate = outgoing[from.0].iter().any(|&arc_id| {
                let existing = &arcs[arc_id.0];
                existing.to == to && existing.arc_type == arc_type
            });
            if duplicate {
                return None;
            }
            let arc_id = ArcId(arcs.len());
            arcs.push(GraphArc {
                id: arc_id,
                from,
                to,
                from_op,
                to_op,
                arc_type,
            });
            outgoing[from.0].push(arc_id);
            incoming[to.0].push(arc_id);
            Some(arc_id)
        };

        // 1. Job precedence arcs in deterministic (job, op_index) order.
        for j in 0..num_jobs {
            let job = JobId(j);
            for k in 0..num_machines.saturating_sub(1) {
                let from_op = OperationId::from_indices(j, k);
                let to_op = OperationId::from_indices(j, k + 1);
                let from_v = VertexId::from_operation(from_op, num_jobs, num_machines)?;
                let to_v = VertexId::from_operation(to_op, num_jobs, num_machines)?;
                if push_arc(
                    &mut arcs,
                    &mut outgoing,
                    &mut incoming,
                    from_v,
                    to_v,
                    Some(from_op),
                    Some(to_op),
                    ArcType::JobPrecedence { job },
                )
                .is_some()
                {
                    num_job_precedence_arcs += 1;
                }
            }
        }

        // 2. Chronological machine-order arcs in deterministic (machine, position) order.
        for (m, seq) in normalized_sequences.iter().enumerate() {
            let machine = MachineId(m);
            for pair in seq.operations.windows(2) {
                let from_op = pair[0];
                let to_op = pair[1];
                let from_v = VertexId::from_operation(from_op, num_jobs, num_machines)?;
                let to_v = VertexId::from_operation(to_op, num_jobs, num_machines)?;
                if let Some(arc_id) = push_arc(
                    &mut arcs,
                    &mut outgoing,
                    &mut incoming,
                    from_v,
                    to_v,
                    Some(from_op),
                    Some(to_op),
                    ArcType::MachineOrder { machine },
                ) {
                    machine_order_arcs[m].push(arc_id);
                    num_machine_order_arcs += 1;
                }
            }
        }

        // Track incoming/outgoing non-boundary counts before adding Source/Sink boundary arcs.
        let has_incoming_constraint: Vec<bool> = incoming
            .iter()
            .take(num_operations + 1)
            .map(|list| !list.is_empty())
            .collect();
        let has_outgoing_constraint: Vec<bool> = outgoing
            .iter()
            .take(num_operations + 1)
            .map(|list| !list.is_empty())
            .collect();

        // 3. Source arcs to operations with no incoming job-precedence or machine-order arc.
        for flat in 0..num_operations {
            let op_v = VertexId(flat + 1);
            if !has_incoming_constraint[op_v.0] {
                let to_op = vertices[op_v.0].operation_id();
                if push_arc(
                    &mut arcs,
                    &mut outgoing,
                    &mut incoming,
                    VertexId::SOURCE,
                    op_v,
                    None,
                    to_op,
                    ArcType::Source,
                )
                .is_some()
                {
                    num_source_arcs += 1;
                }
            }
        }

        // 4. Sink arcs from operations with no outgoing job-precedence or machine-order arc.
        for flat in 0..num_operations {
            let op_v = VertexId(flat + 1);
            if !has_outgoing_constraint[op_v.0] {
                let from_op = vertices[op_v.0].operation_id();
                if push_arc(
                    &mut arcs,
                    &mut outgoing,
                    &mut incoming,
                    op_v,
                    sink_vertex,
                    from_op,
                    None,
                    ArcType::Sink,
                )
                .is_some()
                {
                    num_sink_arcs += 1;
                }
            }
        }

        let metadata = GraphMetadata {
            num_jobs,
            num_machines,
            num_operations,
            num_vertices,
            num_arcs: arcs.len(),
            num_job_precedence_arcs,
            num_machine_order_arcs,
            num_source_arcs,
            num_sink_arcs,
            schedule_makespan,
        };

        Ok(Self {
            metadata,
            vertices,
            arcs,
            outgoing,
            incoming,
            machine_sequences: normalized_sequences,
            machine_order_arcs,
        })
    }

    /// Graph summary counts and schedule makespan.
    #[must_use]
    pub const fn metadata(&self) -> &GraphMetadata {
        &self.metadata
    }

    /// Canonical identifier of the `Source` vertex.
    #[must_use]
    pub const fn source_vertex_id(&self) -> VertexId {
        VertexId::SOURCE
    }

    /// Canonical identifier of the `Sink` vertex.
    #[must_use]
    pub const fn sink_vertex_id(&self) -> VertexId {
        VertexId(self.metadata.num_operations + 1)
    }

    /// All graph vertices in deterministic order (`Source`, operations in `(job, op_index)`
    /// order, `Sink`).
    #[must_use]
    pub fn vertices(&self) -> &[GraphVertex] {
        &self.vertices
    }

    /// Looks up vertex metadata by [`VertexId`].
    ///
    /// # Errors
    /// Returns [`JsspError::InvalidDisjunctiveGraph`] if `id` is out of bounds.
    pub fn vertex(&self, id: VertexId) -> Result<&GraphVertex, JsspError> {
        self.vertices
            .get(id.0)
            .ok_or_else(|| JsspError::InvalidDisjunctiveGraph {
                detail: format!("vertex id {} is out of bounds", id.0),
            })
    }

    /// Resolves `op_id` to its [`VertexId`].
    ///
    /// # Errors
    /// Returns [`JsspError`] if `op_id` is out of bounds for this instance.
    pub fn operation_vertex_id(&self, op_id: OperationId) -> Result<VertexId, JsspError> {
        VertexId::from_operation(op_id, self.metadata.num_jobs, self.metadata.num_machines)
    }

    /// All directed arcs in deterministic construction order.
    #[must_use]
    pub fn arcs(&self) -> &[GraphArc] {
        &self.arcs
    }

    /// Looks up arc metadata by [`ArcId`].
    ///
    /// # Errors
    /// Returns [`JsspError::InvalidDisjunctiveGraph`] if `id` is out of bounds.
    pub fn arc(&self, id: ArcId) -> Result<&GraphArc, JsspError> {
        self.arcs
            .get(id.0)
            .ok_or_else(|| JsspError::InvalidDisjunctiveGraph {
                detail: format!("arc id {} is out of bounds", id.0),
            })
    }

    /// Deterministically ordered outgoing [`ArcId`] slice for vertex `id`.
    ///
    /// # Errors
    /// Returns [`JsspError::InvalidDisjunctiveGraph`] if `id` is out of bounds.
    pub fn outgoing_arcs(&self, id: VertexId) -> Result<&[ArcId], JsspError> {
        self.outgoing.get(id.0).map(Vec::as_slice).ok_or_else(|| {
            JsspError::InvalidDisjunctiveGraph {
                detail: format!("vertex id {} is out of bounds", id.0),
            }
        })
    }

    /// Deterministically ordered incoming [`ArcId`] slice for vertex `id`.
    ///
    /// # Errors
    /// Returns [`JsspError::InvalidDisjunctiveGraph`] if `id` is out of bounds.
    pub fn incoming_arcs(&self, id: VertexId) -> Result<&[ArcId], JsspError> {
        self.incoming.get(id.0).map(Vec::as_slice).ok_or_else(|| {
            JsspError::InvalidDisjunctiveGraph {
                detail: format!("vertex id {} is out of bounds", id.0),
            }
        })
    }

    /// Preserved chronological machine operation sequences from construction.
    #[must_use]
    pub fn machine_sequences(&self) -> &[MachineSequence] {
        &self.machine_sequences
    }

    /// Chronological [`ArcType::MachineOrder`] arc IDs for `machine`.
    ///
    /// # Errors
    /// Returns [`JsspError::InvalidDisjunctiveGraph`] if `machine` is out of bounds.
    pub fn machine_order_arcs(&self, machine: MachineId) -> Result<&[ArcId], JsspError> {
        self.machine_order_arcs
            .get(machine.0)
            .map(Vec::as_slice)
            .ok_or_else(|| JsspError::InvalidDisjunctiveGraph {
                detail: format!("machine id {} is out of bounds", machine.0),
            })
    }

    /// Computes a deterministic topological ordering of all vertices using Kahn's algorithm
    /// with a min-heap keyed by [`VertexId`] (which preserves `Source` first, canonical
    /// `(job, op_index)` tie-breaking among ready operation vertices, and `Sink` last).
    ///
    /// Complexity: `O((V + E) log V)` time, `O(V)` auxiliary space.
    ///
    /// # Errors
    /// Returns [`JsspError::GraphCycleDetected`] if the graph contains a directed cycle,
    /// including a concrete cycle witness of `(job, op_index)` operations.
    pub fn topological_sort(&self) -> Result<Vec<VertexId>, JsspError> {
        let num_vertices = self.vertices.len();
        if num_vertices < 2 {
            return Err(JsspError::InvalidDisjunctiveGraph {
                detail: format!(
                    "disjunctive graph has {num_vertices} vertices, expected at least Source and Sink"
                ),
            });
        }

        let mut in_degree: Vec<usize> = self.incoming.iter().map(Vec::len).collect();
        let mut ready: BinaryHeap<Reverse<VertexId>> = BinaryHeap::new();
        for (idx, &deg) in in_degree.iter().enumerate() {
            if deg == 0 {
                ready.push(Reverse(VertexId(idx)));
            }
        }

        let mut order = Vec::with_capacity(num_vertices);
        let mut visited = vec![false; num_vertices];

        while let Some(Reverse(u)) = ready.pop() {
            visited[u.0] = true;
            order.push(u);
            for &arc_id in &self.outgoing[u.0] {
                let to = self.arcs[arc_id.0].to;
                in_degree[to.0] -= 1;
                if in_degree[to.0] == 0 {
                    ready.push(Reverse(to));
                }
            }
        }

        if order.len() < num_vertices {
            let unvisited_vertices = num_vertices - order.len();
            let cycle_operations = self.extract_cycle_witness(&visited);
            return Err(JsspError::GraphCycleDetected {
                unvisited_vertices,
                cycle_operations,
            });
        }

        // Verify source and sink invariants on valid DAG output.
        let sink_id = self.sink_vertex_id();
        if order.first().copied() != Some(VertexId::SOURCE)
            || order.last().copied() != Some(sink_id)
        {
            return Err(JsspError::InvalidDisjunctiveGraph {
                detail: "topological order violated Source-first / Sink-last boundary invariant"
                    .to_owned(),
            });
        }

        Ok(order)
    }

    fn extract_cycle_witness(&self, visited: &[bool]) -> Vec<(usize, usize)> {
        let start = match visited.iter().enumerate().find(|&(_, &v)| !v) {
            Some((idx, _)) => VertexId(idx),
            None => return Vec::new(),
        };

        // Because Kahn's algorithm terminated with `in_degree[u] > 0` for every unvisited `u`,
        // each unvisited vertex has at least one unvisited predecessor. Walking backward from
        // `start` along minimum unvisited predecessors is guaranteed to repeat a vertex inside
        // a directed cycle.
        let mut path = Vec::new();
        let mut path_pos: HashMap<VertexId, usize> = HashMap::new();
        let mut current = start;

        loop {
            if let Some(&loop_start) = path_pos.get(&current) {
                let mut cycle: Vec<VertexId> = path[loop_start..].to_vec();
                // Reverse backward trace into forward directed arc traversal.
                cycle.reverse();
                // Rotate so the smallest VertexId begins the witness deterministically.
                if let Some((min_pos, _)) = cycle.iter().enumerate().min_by_key(|&(_, v)| v.0) {
                    cycle.rotate_left(min_pos);
                }
                return cycle
                    .into_iter()
                    .filter_map(|v_id| match self.vertices[v_id.0].kind {
                        GraphVertexKind::Operation(op_id) => Some((op_id.job.0, op_id.op_index.0)),
                        GraphVertexKind::Source | GraphVertexKind::Sink => None,
                    })
                    .collect();
            }

            path_pos.insert(current, path.len());
            path.push(current);

            let next_pred = self.incoming[current.0]
                .iter()
                .map(|&arc_id| self.arcs[arc_id.0].from)
                .filter(|from| !visited[from.0])
                .min();

            match next_pred {
                Some(pred) => current = pred,
                None => return Vec::new(),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deliberate_malformed_cycle_via_unchecked_builder_is_detected() {
        // 2x2 cyclic orientation: J0: M0 -> M1, J1: M1 -> M0
        // M0: [O(1,1), O(0,0)], M1: [O(0,1), O(1,0)] creates 4-cycle:
        // O(0,0) -> O(0,1) -> O(1,0) -> O(1,1) -> O(0,0)
        let instance =
            JsspInstance::new(vec![vec![(0, 2), (1, 3)], vec![(1, 4), (0, 1)]]).expect("2x2");
        let seqs = vec![
            MachineSequence::new(
                MachineId(0),
                vec![
                    OperationId::from_indices(1, 1),
                    OperationId::from_indices(0, 0),
                ],
            ),
            MachineSequence::new(
                MachineId(1),
                vec![
                    OperationId::from_indices(0, 1),
                    OperationId::from_indices(1, 0),
                ],
            ),
        ];

        let graph =
            DisjunctiveGraph::from_machine_sequences_unchecked(&instance, &seqs, 0).expect("graph");
        let err = graph.topological_sort().expect_err("cycle must fail");
        assert_eq!(
            err,
            JsspError::GraphCycleDetected {
                unvisited_vertices: 4,
                cycle_operations: vec![(0, 0), (0, 1), (1, 0), (1, 1)],
            }
        );
    }
}
