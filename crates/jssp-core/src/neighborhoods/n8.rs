//! `N8` intra-block + extra-block critical insertion neighborhood (Xie, Li, Gao & Gui, 2023).

use crate::local_search::evaluation::{EvaluatedGraphState, FastGraphEvaluator};
use crate::neighborhoods::moves::{GraphMove, MoveKind, NeighborhoodId};
use crate::neighborhoods::validation::{
    GeneratedNeighborhood, MoveCollector, emit_n6_block_moves, emit_n7_interior_moves,
};

/// Proposition 2 (Xie et al. 2023) backward feasibility for moving `v` before `u`:
/// `L(0, u) + p_u >= L(0, JP[v])`.
#[inline]
#[must_use]
fn is_n8_before_feasible(
    eval: &FastGraphEvaluator,
    state: &EvaluatedGraphState,
    u: u32,
    v: u32,
) -> bool {
    let lhs = state.heads[u as usize] + eval.proc_time(u);
    match eval.job_pred(v) {
        None => true,
        Some(jp_v) => lhs >= state.heads[jp_v as usize],
    }
}

/// Proposition 1 (Xie et al. 2023) forward feasibility for moving `u` after `v`:
/// `L(v, Sink) >= L(JS[u], Sink) - p_{JS[u]} = q_{JS[u]}`.
#[inline]
#[must_use]
fn is_n8_after_feasible(
    eval: &FastGraphEvaluator,
    state: &EvaluatedGraphState,
    u: u32,
    v: u32,
) -> bool {
    let lhs = eval.proc_time(v) + state.tails[v as usize];
    match eval.job_succ(u) {
        None => true,
        Some(js_u) => lhs >= state.tails[js_u as usize],
    }
}

/// Enumerates deduplicated `N8` moves (`N7` intra-block + Propositions 1–2 extra-block shifts)
/// in deterministic critical-block scan order. Every candidate move is also checked via exact
/// DAG cycle detection before acceptance in search.
#[must_use]
pub fn generate_n8(
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

        let full_seq = eval.machine_seq(block.machine);
        let head_pos = eval.mach_pos(block.operations[0]);
        let tail_pos = eval.mach_pos(block.operations[block.operations.len() - 1]);

        // Extra-block backward shifts: insert v in B_l before non-block u scheduled prior to B_l
        for &v in &block.operations {
            let pos_v = eval.mach_pos(v);
            for (pos_u, &u) in full_seq.iter().enumerate().take(head_pos) {
                let kind = if pos_u + 1 == pos_v {
                    MoveKind::AdjacentSwap
                } else {
                    MoveKind::InsertBefore
                };
                let feasible = is_n8_before_feasible(eval, state, u, v);
                collector.push_checked(
                    GraphMove {
                        machine: block.machine,
                        from_pos: pos_v,
                        to_pos: pos_u,
                        moved_op: v,
                        anchor_op: u,
                        kind,
                    },
                    feasible,
                );
            }
        }

        // Extra-block forward shifts: insert u in B_l after non-block v scheduled after B_l
        for &u in &block.operations {
            let pos_u = eval.mach_pos(u);
            for (pos_v, &v) in full_seq.iter().enumerate().skip(tail_pos + 1) {
                let kind = if pos_u + 1 == pos_v {
                    MoveKind::AdjacentSwap
                } else {
                    MoveKind::InsertAfter
                };
                let feasible = is_n8_after_feasible(eval, state, u, v);
                collector.push_checked(
                    GraphMove {
                        machine: block.machine,
                        from_pos: pos_u,
                        to_pos: pos_v,
                        moved_op: u,
                        anchor_op: v,
                        kind,
                    },
                    feasible,
                );
            }
        }
    }

    collector.into_result(NeighborhoodId::N8)
}
