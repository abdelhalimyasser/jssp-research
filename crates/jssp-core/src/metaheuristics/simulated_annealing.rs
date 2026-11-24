//! Simulated Annealing (`B21`–`B24`) on critical-block neighborhoods (`D04`, `D05`).

use crate::error::JsspError;
use crate::initialization::qmc::u01_from_u64;
use crate::local_search::descent::{
    SearchCounters, SearchTerminationReason, checked_probe_move, generate_and_count,
};
use crate::local_search::evaluation::FastGraphEvaluator;
use crate::neighborhoods::NeighborhoodId;
use crate::rng::RunRngStreams;
use crate::telemetry::EvalCounter;

const COOLING_RATE: f64 = 0.95;
const T_MIN: f64 = 1e-3;
const INITIAL_CALIBRATION_PROPOSALS: usize = 50;

/// Executes Simulated Annealing with geometric cooling (`gamma = 0.95`, epoch `2*L`,
/// `T_0` calibrated for ~80% uphill acceptance on first 50 proposals) on `neighborhood`.
///
/// # Errors
/// Returns [`JsspError`] on DAG corruption, RNG error, or arithmetic overflow.
#[allow(clippy::too_many_arguments)]
pub fn run_simulated_annealing(
    evaluator: &mut FastGraphEvaluator,
    neighborhood: NeighborhoodId,
    counter: &mut EvalCounter,
    counters: &mut SearchCounters,
    streams: &mut RunRngStreams,
    best_makespan: &mut u64,
    best_found_eval: &mut usize,
    best_seqs: &mut Vec<Vec<u32>>,
    best_milestones: &mut Vec<(usize, u64)>,
) -> Result<(u64, SearchTerminationReason), JsspError> {
    let Some(mut curr_state) = evaluator.analyze_current()? else {
        return Err(JsspError::InvalidDisjunctiveGraph {
            detail: "Simulated Annealing received cyclic starting graph".to_owned(),
        });
    };

    // 1. Calibration sample for T_0 = max(1.0, -mean(Delta^+) / ln(0.80))
    let mut uphill_sum = 0.0_f64;
    let mut uphill_count = 0usize;
    {
        let gen_init = generate_and_count(evaluator, &curr_state, neighborhood, counters);
        if !gen_init.moves.is_empty() {
            let n_sample = INITIAL_CALIBRATION_PROPOSALS.min(counter.remaining());
            for _ in 0..n_sample {
                let idx = streams
                    .local_search_sample
                    .gen_range_usize(gen_init.moves.len())?;
                let mv = gen_init.moves[idx];
                let Some(probe_res) = checked_probe_move(evaluator, &mv, counter, counters)? else {
                    return Ok((
                        curr_state.makespan,
                        SearchTerminationReason::BudgetExhausted,
                    ));
                };
                if let Some(cand_cmax) = probe_res {
                    if cand_cmax > curr_state.makespan {
                        uphill_sum += (cand_cmax - curr_state.makespan) as f64;
                        uphill_count += 1;
                    } else if cand_cmax < curr_state.makespan {
                        counters.improving_moves += 1;
                    }
                }
            }
        }
    }

    let mut temp = if uphill_count > 0 {
        let mean_delta = uphill_sum / (uphill_count as f64);
        (-mean_delta / 0.80_f64.ln()).max(1.0)
    } else {
        (0.05 * (curr_state.makespan as f64)).max(1.0)
    };

    let epoch_len = (2 * evaluator.num_ops()).max(1);
    let mut epoch_steps = 0usize;

    while counter.remaining() > 0 {
        let gen_n = generate_and_count(evaluator, &curr_state, neighborhood, counters);
        if gen_n.moves.is_empty() {
            return Ok((
                curr_state.makespan,
                SearchTerminationReason::EmptyNeighborhood,
            ));
        }

        let pick = streams
            .local_search_sample
            .gen_range_usize(gen_n.moves.len())?;
        let mv = gen_n.moves[pick];
        let Some(probe_res) = checked_probe_move(evaluator, &mv, counter, counters)? else {
            return Ok((
                curr_state.makespan,
                SearchTerminationReason::BudgetExhausted,
            ));
        };

        epoch_steps += 1;
        if let Some(cand_cmax) = probe_res {
            let accept = if cand_cmax <= curr_state.makespan {
                if cand_cmax < curr_state.makespan {
                    counters.improving_moves += 1;
                }
                true
            } else {
                let delta = (cand_cmax - curr_state.makespan) as f64;
                let prob = (-delta / temp).exp();
                u01_from_u64(streams.sa_accept.next_u64()) < prob
            };

            if accept {
                evaluator.commit_move(&mv);
                counters.iterations += 1;
                counters.accepted_moves += 1;
                curr_state = evaluator.analyze_current()?.ok_or_else(|| {
                    JsspError::InvalidDisjunctiveGraph {
                        detail: "accepted SA move induced unexpected cycle".to_owned(),
                    }
                })?;
                if curr_state.makespan < *best_makespan {
                    *best_makespan = curr_state.makespan;
                    *best_found_eval = counter.total();
                    *best_seqs = evaluator.snapshot_machine_seqs();
                    best_milestones.push((*best_found_eval, curr_state.makespan));
                }
            }
        }

        if epoch_steps >= epoch_len {
            epoch_steps = 0;
            temp = (temp * COOLING_RATE).max(T_MIN);
        }
    }

    Ok((
        curr_state.makespan,
        SearchTerminationReason::BudgetExhausted,
    ))
}
