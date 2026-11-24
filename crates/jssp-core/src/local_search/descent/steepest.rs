//! Deterministic Steepest Descent local search (`B01`–`B04`) (`D05`).

use crate::error::JsspError;
use crate::local_search::descent::{
    SearchCounters, SearchTerminationReason, checked_probe_move, generate_and_count,
};
use crate::local_search::evaluation::FastGraphEvaluator;
use crate::neighborhoods::{GraphMove, NeighborhoodId};
use crate::telemetry::EvalCounter;

/// Executes Steepest Descent from `evaluator`'s current machine-sequence state within
/// `counter`, updating global `best_makespan`, `best_found_eval`, `best_seqs`, and
/// `best_milestones`.
///
/// Only sets [`SearchTerminationReason::LocalOptimum`] if an entire deduplicated neighborhood
/// completes evaluation with zero strictly improving moves; exhausting `counter` mid-scan
/// returns [`SearchTerminationReason::BudgetExhausted`].
///
/// # Errors
/// Returns [`JsspError`] on DAG corruption or arithmetic overflow.
#[allow(clippy::too_many_arguments)]
pub fn run_steepest_descent(
    evaluator: &mut FastGraphEvaluator,
    neighborhood: NeighborhoodId,
    counter: &mut EvalCounter,
    counters: &mut SearchCounters,
    best_makespan: &mut u64,
    best_found_eval: &mut usize,
    best_seqs: &mut Vec<Vec<u32>>,
    best_milestones: &mut Vec<(usize, u64)>,
) -> Result<(u64, SearchTerminationReason), JsspError> {
    let Some(mut curr_state) = evaluator.analyze_current()? else {
        return Err(JsspError::InvalidDisjunctiveGraph {
            detail: "Steepest Descent received cyclic starting graph".to_owned(),
        });
    };
    let mut curr_cmax = curr_state.makespan;

    loop {
        if counter.remaining() == 0 {
            return Ok((curr_cmax, SearchTerminationReason::BudgetExhausted));
        }

        let gen_n = generate_and_count(evaluator, &curr_state, neighborhood, counters);
        if gen_n.moves.is_empty() {
            return Ok((curr_cmax, SearchTerminationReason::LocalOptimum));
        }

        let mut best_cand: Option<(GraphMove, u64)> = None;
        let mut completed_full_scan = true;

        for &mv in &gen_n.moves {
            let Some(probe_res) = checked_probe_move(evaluator, &mv, counter, counters)? else {
                completed_full_scan = false;
                break;
            };
            if let Some(cand_cmax) = probe_res {
                if cand_cmax < curr_cmax {
                    counters.improving_moves += 1;
                }
                let is_better = match best_cand {
                    None => cand_cmax < curr_cmax,
                    Some((_, bc)) => cand_cmax < bc,
                };
                if is_better {
                    best_cand = Some((mv, cand_cmax));
                }
            }
        }

        if let Some((best_mv, new_cmax)) = best_cand
            && new_cmax < curr_cmax
        {
            evaluator.commit_move(&best_mv);
            counters.iterations += 1;
            counters.accepted_moves += 1;
            curr_state =
                evaluator
                    .analyze_current()?
                    .ok_or_else(|| JsspError::InvalidDisjunctiveGraph {
                        detail: "committed Steepest Descent move produced unexpected cycle"
                            .to_owned(),
                    })?;
            curr_cmax = curr_state.makespan;
            if curr_cmax < *best_makespan {
                *best_makespan = curr_cmax;
                *best_found_eval = counter.total();
                *best_seqs = evaluator.snapshot_machine_seqs();
                best_milestones.push((*best_found_eval, curr_cmax));
            }
            if !completed_full_scan {
                return Ok((curr_cmax, SearchTerminationReason::BudgetExhausted));
            }
        } else if !completed_full_scan {
            return Ok((curr_cmax, SearchTerminationReason::BudgetExhausted));
        } else {
            return Ok((curr_cmax, SearchTerminationReason::LocalOptimum));
        }
    }
}
