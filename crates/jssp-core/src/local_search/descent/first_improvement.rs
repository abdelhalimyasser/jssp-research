//! Deterministic First Improvement local search (`B05`–`B08`) (`D05`).

use crate::error::JsspError;
use crate::local_search::descent::{
    SearchCounters, SearchTerminationReason, checked_probe_move, generate_and_count,
};
use crate::local_search::evaluation::FastGraphEvaluator;
use crate::neighborhoods::NeighborhoodId;
use crate::telemetry::EvalCounter;

/// Executes First Improvement local search scanning candidate moves in canonical
/// deterministic critical-block order and accepting the first strict improvement.
///
/// # Errors
/// Returns [`JsspError`] on DAG corruption or arithmetic overflow.
#[allow(clippy::too_many_arguments)]
pub fn run_first_improvement(
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
            detail: "First Improvement received cyclic starting graph".to_owned(),
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

        let mut improved = false;
        let mut budget_hit = false;

        for &mv in &gen_n.moves {
            let Some(probe_res) = checked_probe_move(evaluator, &mv, counter, counters)? else {
                budget_hit = true;
                break;
            };
            if let Some(cand_cmax) = probe_res
                && cand_cmax < curr_cmax
            {
                counters.improving_moves += 1;
                counters.accepted_moves += 1;
                counters.iterations += 1;
                evaluator.commit_move(&mv);
                curr_state = evaluator.analyze_current()?.ok_or_else(|| {
                    JsspError::InvalidDisjunctiveGraph {
                        detail: "committed First Improvement move induced cycle".to_owned(),
                    }
                })?;
                curr_cmax = curr_state.makespan;
                if curr_cmax < *best_makespan {
                    *best_makespan = curr_cmax;
                    *best_found_eval = counter.total();
                    *best_seqs = evaluator.snapshot_machine_seqs();
                    best_milestones.push((*best_found_eval, curr_cmax));
                }
                improved = true;
                break;
            }
        }

        if budget_hit {
            return Ok((curr_cmax, SearchTerminationReason::BudgetExhausted));
        }
        if !improved {
            return Ok((curr_cmax, SearchTerminationReason::LocalOptimum));
        }
    }
}
