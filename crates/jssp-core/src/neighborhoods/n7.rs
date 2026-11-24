//! `N7` extended intra-block neighborhood (Zhang, Li, Guan & Rao, 2007).

use crate::local_search::evaluation::{EvaluatedGraphState, FastGraphEvaluator};
use crate::neighborhoods::moves::NeighborhoodId;
use crate::neighborhoods::validation::{
    GeneratedNeighborhood, MoveCollector, emit_n6_block_moves, emit_n7_interior_moves,
};

/// Enumerates deduplicated, head/tail-feasible `N7` moves (`N6` + endpoint-into-interior)
/// in deterministic critical-block scan order.
#[must_use]
pub fn generate_n7(
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
        emit_n7_interior_moves(
            eval,
            state,
            &block.operations,
            block.machine,
            is_first,
            is_last,
            &mut collector,
        );
    }

    collector.into_result(NeighborhoodId::N7)
}
