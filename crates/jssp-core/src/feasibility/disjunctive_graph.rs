//! Disjunctive-graph acyclicity verification facade.
//!
//! All disjunctive-graph data structures and topological cycle detection live in
//! [`crate::graph`]. This module exposes a clean feasibility check wrapper so there is
//! strictly one source of truth for graph semantics.

use crate::error::JsspError;
use crate::graph::{DisjunctiveGraph, VertexId};

/// Verifies that the oriented disjunctive graph `graph` is a directed acyclic graph (DAG)
/// and returns its deterministic topological vertex order.
///
/// # Errors
/// Returns [`JsspError::GraphCycleDetected`] if the graph contains a directed cycle, or
/// [`JsspError::InvalidDisjunctiveGraph`] if structural invariants are violated.
pub fn check_disjunctive_graph_acyclicity(
    graph: &DisjunctiveGraph,
) -> Result<Vec<VertexId>, JsspError> {
    graph.topological_sort()
}
