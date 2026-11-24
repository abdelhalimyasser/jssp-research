//! `N6` critical-block endpoint insertion neighborhood (Balas & Vazacopoulos, 1998).

use crate::local_search::evaluation::{EvaluatedGraphState, FastGraphEvaluator};
use crate::neighborhoods::moves::NeighborhoodId;
use crate::neighborhoods::validation::{GeneratedNeighborhood, MoveCollector, emit_n6_block_moves};

/// Enumerates deduplicated, head/tail-feasible `N6` moves in deterministic scan order.
#[must_use]
pub fn generate_n6(
    eval: &FastGraphEvaluator,
    state: &EvaluatedGraphState,
) -> GeneratedNeighborhood {
    let mut collector = MoveCollector::default();

    for block in &state.blocks {
        let is_first = block.block_index == 0;
        let is_last = block.block_index + 1 == block.total_blocks;
        emit_n6_block_moves(
            eval,
            state,
            &block.operations,
            block.machine,
            is_first,
            is_last,
            &mut collector,
        );
    }

    collector.into_result(NeighborhoodId::N6)
}
