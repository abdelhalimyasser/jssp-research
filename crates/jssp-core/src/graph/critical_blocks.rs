//! Maximal critical block extraction on machine operation sequences.
//!
//! A **critical block** is a maximal contiguous sequence of operations `(b_1, ..., b_k)`
//! (`k >= 2`) processed sequentially on the same machine such that every consecutive
//! [`ArcType::MachineOrder`] arc `b_i -> b_{i+1}` belongs to at least one critical
//! `Source`-to-`Sink` path.
//!
//! Two adjacent critical operations (`slack == 0`) that are connected by a **non-critical**
//! machine-order arc (`head[u] + p[u] < head[v]`) are **not** merged into one critical block.

use std::fmt;

use crate::graph::critical_path::AnalyzedArc;
use crate::graph::{ArcId, DisjunctiveGraph};
use crate::model::{MachineId, OperationId};

/// Strongly typed 0-based identifier for an extracted [`CriticalBlock`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CriticalBlockId(pub usize);

impl CriticalBlockId {
    /// Constructs a `CriticalBlockId` from a zero-based block index.
    #[must_use]
    pub const fn new(index: usize) -> Self {
        Self(index)
    }

    /// Returns the underlying zero-based block index.
    #[must_use]
    pub const fn index(self) -> usize {
        self.0
    }
}

impl From<usize> for CriticalBlockId {
    fn from(value: usize) -> Self {
        Self(value)
    }
}

impl fmt::Display for CriticalBlockId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "B{}", self.0)
    }
}

/// Maximal contiguous run of `>= 2` operations on one machine connected by critical
/// machine-order arcs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CriticalBlock {
    /// Stable deterministic block identifier (`0..num_blocks` in machine-major chronological order).
    pub id: CriticalBlockId,
    /// Machine processing every operation in this block.
    pub machine: MachineId,
    /// Chronologically ordered operations inside this block (`len >= 2`).
    pub operations: Vec<OperationId>,
    /// First operation `b_1` of the block.
    pub first_operation: OperationId,
    /// Last operation `b_k` of the block.
    pub last_operation: OperationId,
    /// Number of operations in this block (`operations.len()`).
    pub len: usize,
    /// Critical machine-order arcs `b_i -> b_{i+1}` (`len == operations.len() - 1`).
    pub critical_machine_arcs: Vec<ArcId>,
}

/// Extracts all maximal critical blocks across every machine in deterministic `(machine,
/// chronological position)` order.
///
/// Complexity: `O(V + E)` linear scan over every machine sequence and its consecutive
/// machine-order arcs.
#[must_use]
pub fn extract_critical_blocks(
    graph: &DisjunctiveGraph,
    analyzed_arcs: &[AnalyzedArc],
) -> Vec<CriticalBlock> {
    let mut blocks = Vec::new();
    let num_machines = graph.metadata().num_machines;

    for m in 0..num_machines {
        let machine = MachineId(m);
        let seq = &graph.machine_sequences()[m].operations;
        let m_arcs = match graph.machine_order_arcs(machine) {
            Ok(arcs) => arcs,
            Err(_) => continue,
        };

        let mut arc_idx = 0usize;
        while arc_idx < m_arcs.len() {
            let arc_id = m_arcs[arc_idx];
            if !analyzed_arcs[arc_id.0].is_critical {
                arc_idx += 1;
                continue;
            }

            let run_start = arc_idx;
            let mut block_arcs = Vec::new();
            while arc_idx < m_arcs.len() && analyzed_arcs[m_arcs[arc_idx].0].is_critical {
                block_arcs.push(m_arcs[arc_idx]);
                arc_idx += 1;
            }

            let run_end_op_idx = arc_idx; // inclusive end operation index (= last_critical_arc + 1)
            let ops = seq[run_start..=run_end_op_idx].to_vec();
            let block_len = ops.len();
            if block_len >= 2 {
                let first_operation = ops[0];
                let last_operation = ops[block_len - 1];
                let id = CriticalBlockId(blocks.len());
                blocks.push(CriticalBlock {
                    id,
                    machine,
                    operations: ops,
                    first_operation,
                    last_operation,
                    len: block_len,
                    critical_machine_arcs: block_arcs,
                });
            }
        }
    }

    blocks
}
