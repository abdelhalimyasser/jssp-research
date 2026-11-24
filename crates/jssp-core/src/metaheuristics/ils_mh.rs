//! Hybrid Metropolis-Filtered ILS Perturbation + Steepest Descent (`B29`–`B32`) (`D04-B`).

use crate::decoder::SerialDecoderWorkspace;
use crate::error::JsspError;
use crate::initialization::qmc::u01_from_u64;
use crate::local_search::descent::{SearchCounters, SearchTerminationReason, run_steepest_descent};
use crate::local_search::evaluation::FastGraphEvaluator;
use crate::metaheuristics::perturb_obr_three_cycle;
use crate::model::JsspInstance;
use crate::neighborhoods::NeighborhoodId;
use crate::representation::ObrChromosome;
use crate::rng::RunRngStreams;
use crate::telemetry::EvalCounter;

/// Executes Metropolis-Filtered ILS Perturbation + Deterministic Steepest Descent (`D04-B`):
///
/// 1. Run deterministic Steepest Descent on `neighborhood`.
/// 2. Sample a symmetric 3-cycle distinct-job OBR perturbation `pi'` (`perturbation` stream).
/// 3. Decode `E(pi') = C_max(Decode(pi'))` (1 eval).
/// 4. Filter via Metropolis acceptance `min(1, exp(-beta_filter * (E(pi') - E(pi))))`
///    with `beta_filter = 2.0 / mean_proc_time` (`mh_filter` stream).
/// 5. Execute Steepest Descent iff accepted by the perturbation filter.
///
/// # Errors
/// Returns [`JsspError`] on DAG failure, RNG error, or arithmetic overflow.
#[allow(clippy::too_many_arguments)]
pub fn run_ils_mh(
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
    let (mut curr_cmax, init_reason) = run_steepest_descent(
        evaluator,
        neighborhood,
        counter,
        counters,
        best_makespan,
        best_found_eval,
        best_seqs,
        best_milestones,
    )?;
    if counter.remaining() == 0 {
        return Ok((curr_cmax, init_reason));
    }

    let mut curr_seqs = evaluator.snapshot_machine_seqs();
    let beta_filter = 2.0 / evaluator.mean_proc_time().max(1.0);

    while counter.remaining() > 0 {
        evaluator.restore_machine_seqs(&curr_seqs);
        let state =
            evaluator
                .analyze_current()?
                .ok_or_else(|| JsspError::InvalidDisjunctiveGraph {
                    detail: "ILS+MH base state became cyclic".to_owned(),
                })?;
        let obr = evaluator.encode_canonical_obr(&state);
        let mut genes = obr.to_job_indices();
        perturb_obr_three_cycle(&mut genes, instance.num_jobs(), &mut streams.perturbation)?;
        counters.perturbations_applied += 1;

        if !counter.try_consume_search() {
            return Ok((curr_cmax, SearchTerminationReason::BudgetExhausted));
        }
        let pert_obr = ObrChromosome::from_job_indices(&genes);
        let pert_sched = decoder_ws.decode(instance, &pert_obr, true)?;
        let e_pert = pert_sched.makespan();

        let filter_accept = if e_pert <= curr_cmax {
            true
        } else {
            let delta = (e_pert - curr_cmax) as f64;
            let alpha_pert = (-beta_filter * delta).exp();
            u01_from_u64(streams.mh_filter.next_u64()) < alpha_pert
        };

        if filter_accept {
            counters.perturbations_accepted += 1;
            evaluator.load_from_schedule(&pert_sched)?;
            if e_pert < *best_makespan {
                *best_makespan = e_pert;
                *best_found_eval = counter.total();
                *best_seqs = evaluator.snapshot_machine_seqs();
                best_milestones.push((*best_found_eval, e_pert));
            }
            let (ls_cmax, _) = run_steepest_descent(
                evaluator,
                neighborhood,
                counter,
                counters,
                best_makespan,
                best_found_eval,
                best_seqs,
                best_milestones,
            )?;
            curr_cmax = ls_cmax;
            curr_seqs = evaluator.snapshot_machine_seqs();
        }
    }

    Ok((curr_cmax, SearchTerminationReason::BudgetExhausted))
}
