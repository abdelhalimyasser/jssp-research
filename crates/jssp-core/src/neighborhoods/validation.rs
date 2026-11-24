//! Common deterministic move generation, head/tail feasibility, clipping & deduplication.

use std::collections::HashSet;

use crate::local_search::evaluation::{EvaluatedGraphState, FastGraphEvaluator};
use crate::neighborhoods::moves::{GraphMove, MoveKind, NeighborhoodId};

/// Result of enumerating one deduplicated disjunctive-graph neighborhood on a DAG state.
#[derive(Debug, Clone)]
pub struct GeneratedNeighborhood {
    pub neighborhood: NeighborhoodId,
    pub moves: Vec<GraphMove>,
    pub raw_generated_count: usize,
    pub duplicate_pruned_count: usize,
    pub infeasible_precond_count: usize,
    pub clipped_boundary_count: usize,
}

#[derive(Debug, Default)]
pub(crate) struct MoveCollector {
    seen: HashSet<(usize, usize, usize, u8)>,
    pub(crate) moves: Vec<GraphMove>,
    pub(crate) raw_generated_count: usize,
    pub(crate) duplicate_pruned_count: usize,
    pub(crate) infeasible_precond_count: usize,
    pub(crate) clipped_boundary_count: usize,
}

impl MoveCollector {
    pub(crate) fn push_checked(&mut self, mv: GraphMove, feasible_precond: bool) {
        self.raw_generated_count += 1;
        if !feasible_precond {
            self.infeasible_precond_count += 1;
            return;
        }
        if mv.from_pos == mv.to_pos {
            return;
        }
        let key = mv.canonical_dedup_key();
        if self.seen.insert(key) {
            self.moves.push(mv);
        } else {
            self.duplicate_pruned_count += 1;
        }
    }

    pub(crate) fn record_clipped(&mut self) {
        self.clipped_boundary_count += 1;
    }

    pub(crate) fn into_result(self, neighborhood: NeighborhoodId) -> GeneratedNeighborhood {
        GeneratedNeighborhood {
            neighborhood,
            moves: self.moves,
            raw_generated_count: self.raw_generated_count,
            duplicate_pruned_count: self.duplicate_pruned_count,
            infeasible_precond_count: self.infeasible_precond_count,
            clipped_boundary_count: self.clipped_boundary_count,
        }
    }
}

/// Exact Balas–Vazacopoulos / Zhang et al. Theorems 1–2 sufficient condition for inserting
/// later operation `v` immediately BEFORE earlier operation `u` on the same machine:
/// `r_u + p_u >= r_{JP[v]} + p_{JP[v]}`.
#[inline]
#[must_use]
pub(crate) fn is_bv_before_feasible(
    eval: &FastGraphEvaluator,
    state: &EvaluatedGraphState,
    u: u32,
    v: u32,
) -> bool {
    let lhs = state.heads[u as usize] + eval.proc_time(u);
    match eval.job_pred(v) {
        None => true,
        Some(jp_v) => lhs >= state.heads[jp_v as usize] + eval.proc_time(jp_v),
    }
}

/// Exact Balas–Vazacopoulos / Zhang et al. Theorems 1–2 sufficient condition for inserting
/// earlier operation `u` immediately AFTER later operation `v` on the same machine:
/// `p_v + q_v >= p_{JS[u]} + q_{JS[u]}`.
#[inline]
#[must_use]
pub(crate) fn is_bv_after_feasible(
    eval: &FastGraphEvaluator,
    state: &EvaluatedGraphState,
    u: u32,
    v: u32,
) -> bool {
    let lhs = eval.proc_time(v) + state.tails[v as usize];
    match eval.job_succ(u) {
        None => true,
        Some(js_u) => lhs >= eval.proc_time(js_u) + state.tails[js_u as usize],
    }
}

/// Emits shared `N6` endpoint shifts for one critical block `B_l = (b_1, ..., b_k)` (`k >= 2`).
pub(crate) fn emit_n6_block_moves(
    eval: &FastGraphEvaluator,
    state: &EvaluatedGraphState,
    block_ops: &[u32],
    machine: usize,
    is_first_block: bool,
    is_last_block: bool,
    collector: &mut MoveCollector,
) {
    let k = block_ops.len();
    if k < 2 {
        return;
    }

    let b1 = block_ops[0];
    let bk = block_ops[k - 1];
    let pos_b1 = eval.mach_pos(b1);
    let pos_bk = eval.mach_pos(bk);

    // Move v in {b_2..b_k} before b_1 (clipped in B_1 by Prop. 3 / Prop. 5)
    for &v in &block_ops[1..k] {
        if is_first_block {
            collector.record_clipped();
            continue;
        }
        let pos_v = eval.mach_pos(v);
        let kind = if pos_v == pos_b1 + 1 {
            MoveKind::AdjacentSwap
        } else {
            MoveKind::InsertBefore
        };
        let feasible = is_bv_before_feasible(eval, state, b1, v);
        collector.push_checked(
            GraphMove {
                machine,
                from_pos: pos_v,
                to_pos: pos_b1,
                moved_op: v,
                anchor_op: b1,
                kind,
            },
            feasible,
        );
    }

    // Move u in {b_1..b_{k-1}} after b_k (clipped in B_b by Prop. 3 / Prop. 6)
    for &u in &block_ops[0..(k - 1)] {
        if is_last_block {
            collector.record_clipped();
            continue;
        }
        let pos_u = eval.mach_pos(u);
        let kind = if pos_u + 1 == pos_bk {
            MoveKind::AdjacentSwap
        } else {
            MoveKind::InsertAfter
        };
        let feasible = is_bv_after_feasible(eval, state, u, bk);
        collector.push_checked(
            GraphMove {
                machine,
                from_pos: pos_u,
                to_pos: pos_bk,
                moved_op: u,
                anchor_op: bk,
                kind,
            },
            feasible,
        );
    }
}

/// Emits additional `N7` endpoint-into-interior shifts for one critical block (`k >= 3`).
pub(crate) fn emit_n7_interior_moves(
    eval: &FastGraphEvaluator,
    state: &EvaluatedGraphState,
    block_ops: &[u32],
    machine: usize,
    is_first_block: bool,
    is_last_block: bool,
    collector: &mut MoveCollector,
) {
    let k = block_ops.len();
    if k < 3 {
        return;
    }

    let b1 = block_ops[0];
    let bk = block_ops[k - 1];
    let pos_b1 = eval.mach_pos(b1);
    let pos_bk = eval.mach_pos(bk);

    // Move b_1 after inner v in {b_2..b_{k-1}} (clipped in B_1 by Xie Prop. 3)
    for &v in &block_ops[1..(k - 1)] {
        if is_first_block {
            collector.record_clipped();
            continue;
        }
        let pos_v = eval.mach_pos(v);
        let kind = if pos_b1 + 1 == pos_v {
            MoveKind::AdjacentSwap
        } else {
            MoveKind::InsertAfter
        };
        let feasible = is_bv_after_feasible(eval, state, b1, v);
        collector.push_checked(
            GraphMove {
                machine,
                from_pos: pos_b1,
                to_pos: pos_v,
                moved_op: b1,
                anchor_op: v,
                kind,
            },
            feasible,
        );
    }

    // Move b_k before inner u in {b_2..b_{k-1}} (clipped in B_b by Xie Prop. 4)
    for &u in &block_ops[1..(k - 1)] {
        if is_last_block {
            collector.record_clipped();
            continue;
        }
        let pos_u = eval.mach_pos(u);
        let kind = if pos_u + 1 == pos_bk {
            MoveKind::AdjacentSwap
        } else {
            MoveKind::InsertBefore
        };
        let feasible = is_bv_before_feasible(eval, state, u, bk);
        collector.push_checked(
            GraphMove {
                machine,
                from_pos: pos_bk,
                to_pos: pos_u,
                moved_op: bk,
                anchor_op: u,
                kind,
            },
            feasible,
        );
    }
}
