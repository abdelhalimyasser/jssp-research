//! Dynamic-tenure Tabu Search (`B13`–`B16`) on critical-block neighborhoods (`D05`).

use std::collections::HashMap;

use crate::decoder::SerialDecoderWorkspace;
use crate::error::JsspError;
use crate::local_search::descent::{
    SearchCounters, SearchTerminationReason, checked_probe_move, generate_and_count,
};
use crate::local_search::evaluation::FastGraphEvaluator;
use crate::metaheuristics::perturb_obr_three_cycle;
use crate::model::JsspInstance;
use crate::neighborhoods::{GraphMove, NeighborhoodId};
use crate::representation::ObrChromosome;
use crate::rng::{NamedRngStream, RunRngStreams};
use crate::telemetry::EvalCounter;

const TABU_STAGNATION_ITERS: usize = 200;

/// Runs dynamic-tenure Tabu Search (`[L_min, 1.5*L_min]`, aspiration, oldest-tabu fallback,
/// and 200-iter 3-cycle OBR diversification) on `neighborhood` within `counter`.
///
/// # Errors
/// Returns [`JsspError`] on DAG failure, RNG error, or arithmetic overflow.
#[allow(clippy::too_many_arguments)]
pub fn run_tabu_search(
    instance: &JsspInstance,
    evaluator: &mut FastGraphEvaluator,
    decoder_ws: &mut SerialDecoderWorkspace,
    neighborhood: NeighborhoodId,
    counter: &mut EvalCounter,
    counters: &mut SearchCounters,
    streams: &mut RunRngStreams,
    best_makespan: &mut u64,
    best_found_eval: &mut usize,
    best_seqs: &mut Vec<Vec<u32>>,
    best_milestones: &mut Vec<(usize, u64)>,
) -> Result<(u64, SearchTerminationReason), JsspError> {
    let l_min = 10 + (instance.num_jobs() / instance.num_machines().max(1));
    let l_max = (3 * l_min) / 2;
    let mut tabu_until: HashMap<(u32, u32), usize> = HashMap::new();
    let mut stagnant_iters = 0usize;
    let mut iter = 0usize;

    let Some(mut curr_state) = evaluator.analyze_current()? else {
        return Err(JsspError::InvalidDisjunctiveGraph {
            detail: "Tabu Search received cyclic starting graph".to_owned(),
        });
    };

    while counter.remaining() > 0 {
        iter += 1;
        let gen_n = generate_and_count(evaluator, &curr_state, neighborhood, counters);
        if gen_n.moves.is_empty() {
            return Ok((
                curr_state.makespan,
                SearchTerminationReason::EmptyNeighborhood,
            ));
        }

        let mut best_admissible: Option<(GraphMove, u64)> = None;
        let mut best_tabu_fallback: Option<(GraphMove, usize, u64)> = None;
        let mut hit_budget = false;

        for &mv in &gen_n.moves {
            let Some(probe_res) = checked_probe_move(evaluator, &mv, counter, counters)? else {
                hit_budget = true;
                break;
            };
            let Some(cand_cmax) = probe_res else {
                continue;
            };
            if cand_cmax < curr_state.makespan {
                counters.improving_moves += 1;
            }
            let (created_order, _) = mv.tabu_orders();
            let expire = tabu_until.get(&created_order).copied().unwrap_or(0);
            let is_tabu = expire > iter;
            let aspirated = cand_cmax < *best_makespan;

            if !is_tabu || aspirated {
                let take = match best_admissible {
                    None => true,
                    Some((_, bc)) => cand_cmax < bc,
                };
                if take {
                    best_admissible = Some((mv, cand_cmax));
                }
            } else {
                let take = match best_tabu_fallback {
                    None => true,
                    Some((_, b_exp, b_c)) => expire < b_exp || (expire == b_exp && cand_cmax < b_c),
                };
                if take {
                    best_tabu_fallback = Some((mv, expire, cand_cmax));
                }
            }
        }

        if let Some((chosen_mv, _)) =
            best_admissible.or_else(|| best_tabu_fallback.map(|(m, _, c)| (m, c)))
        {
            let (_, broken_order) = chosen_mv.tabu_orders();
            evaluator.commit_move(&chosen_mv);
            counters.iterations += 1;
            counters.accepted_moves += 1;

            let tenure = sample_dynamic_tenure(l_min, l_max, &mut streams.local_search_sample)?;
            tabu_until.insert(broken_order, iter + tenure);

            curr_state =
                evaluator
                    .analyze_current()?
                    .ok_or_else(|| JsspError::InvalidDisjunctiveGraph {
                        detail: "committed Tabu Search move produced cycle".to_owned(),
                    })?;

            if curr_state.makespan < *best_makespan {
                *best_makespan = curr_state.makespan;
                *best_found_eval = counter.total();
                *best_seqs = evaluator.snapshot_machine_seqs();
                best_milestones.push((*best_found_eval, curr_state.makespan));
                stagnant_iters = 0;
            } else {
                stagnant_iters += 1;
            }
        }

        if hit_budget || counter.remaining() == 0 {
            return Ok((
                curr_state.makespan,
                SearchTerminationReason::BudgetExhausted,
            ));
        }

        if stagnant_iters >= TABU_STAGNATION_ITERS && counter.remaining() > 0 {
            let obr = evaluator.encode_canonical_obr(&curr_state);
            let mut genes = obr.to_job_indices();
            perturb_obr_three_cycle(&mut genes, instance.num_jobs(), &mut streams.perturbation)?;
            if !counter.try_consume_search() {
                return Ok((
                    curr_state.makespan,
                    SearchTerminationReason::BudgetExhausted,
                ));
            }
            let pert_obr = ObrChromosome::from_job_indices(&genes);
            let pert_sched = decoder_ws.decode(instance, &pert_obr, true)?;
            evaluator.load_from_schedule(&pert_sched)?;
            curr_state =
                evaluator
                    .analyze_current()?
                    .ok_or_else(|| JsspError::InvalidDisjunctiveGraph {
                        detail: "M2 decoded Tabu diversification produced cycle".to_owned(),
                    })?;
            counters.perturbations_applied += 1;
            counters.restarts_triggered += 1;
            tabu_until.clear();
            stagnant_iters = 0;
            if curr_state.makespan < *best_makespan {
                *best_makespan = curr_state.makespan;
                *best_found_eval = counter.total();
                *best_seqs = evaluator.snapshot_machine_seqs();
                best_milestones.push((*best_found_eval, curr_state.makespan));
            }
        }
    }

    Ok((
        curr_state.makespan,
        SearchTerminationReason::BudgetExhausted,
    ))
}

fn sample_dynamic_tenure(
    l_min: usize,
    l_max: usize,
    rng: &mut NamedRngStream,
) -> Result<usize, JsspError> {
    if l_max <= l_min {
        return Ok(l_min);
    }
    Ok(l_min + rng.gen_range_usize(l_max - l_min + 1)?)
}
