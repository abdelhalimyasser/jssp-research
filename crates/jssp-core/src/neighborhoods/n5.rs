//! `N5` critical-block boundary adjacent-swap neighborhood (Nowicki & Smutnicki, 1996).

use crate::local_search::evaluation::{EvaluatedGraphState, FastGraphEvaluator};
use crate::neighborhoods::moves::{GraphMove, MoveKind, NeighborhoodId};
use crate::neighborhoods::validation::{GeneratedNeighborhood, MoveCollector};

/// Enumerates deduplicated `N5` moves in deterministic critical-block scan order.
#[must_use]
pub fn generate_n5(
    eval: &FastGraphEvaluator,
    state: &EvaluatedGraphState,
) -> GeneratedNeighborhood {
    let mut collector = MoveCollector::default();

    for block in &state.blocks {
        let k = block.operations.len();
        if k < 2 {
            continue;
        }
        let is_first = block.block_index == 0;
        let is_last = block.block_index + 1 == block.total_blocks;

        let b1 = block.operations[0];
        let b2 = block.operations[1];
        if is_first {
            collector.record_clipped();
        } else {
            let p1 = eval.mach_pos(b1);
            let p2 = eval.mach_pos(b2);
            collector.push_checked(
                GraphMove {
                    machine: block.machine,
                    from_pos: p2,
                    to_pos: p1,
                    moved_op: b2,
                    anchor_op: b1,
                    kind: MoveKind::AdjacentSwap,
                },
                true,
            );
        }

        let bk_prev = block.operations[k - 2];
        let bk = block.operations[k - 1];
        if is_last {
            collector.record_clipped();
        } else {
            let p_prev = eval.mach_pos(bk_prev);
            let p_last = eval.mach_pos(bk);
            collector.push_checked(
                GraphMove {
                    machine: block.machine,
                    from_pos: p_prev,
                    to_pos: p_last,
                    moved_op: bk_prev,
                    anchor_op: bk,
                    kind: MoveKind::AdjacentSwap,
                },
                true,
            );
        }
    }

    collector.into_result(NeighborhoodId::N5)
}
