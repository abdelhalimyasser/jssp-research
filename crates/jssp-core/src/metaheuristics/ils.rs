//! Iterated Local Search (`B17`–`B20`) with Steepest Descent & 3-cycle OBR perturbation (`D05`).

use crate::decoder::SerialDecoderWorkspace;
use crate::error::JsspError;
use crate::local_search::descent::{SearchCounters, SearchTerminationReason, run_steepest_descent};
use crate::local_search::evaluation::FastGraphEvaluator;
use crate::metaheuristics::perturb_obr_three_cycle;
use crate::model::JsspInstance;
use crate::neighborhoods::NeighborhoodId;
use crate::representation::ObrChromosome;
use crate::rng::RunRngStreams;
use crate::telemetry::EvalCounter;

const ILS_STAGNATION_PERTS: usize = 20;

/// Executes Iterated Local Search pairing Steepest Descent (`B01`–`B04`) with 3-cycle
/// distinct-job OBR perturbation (`D04-B` Option B), accepting `<= curr` and restarting
/// from `best_seqs` after 20 stagnant perturbations.
///
/// # Errors
/// Returns [`JsspError`] on DAG corruption, RNG error, or arithmetic overflow.
#[allow(clippy::too_many_arguments)]
pub fn run_ils(
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
    let mut stagnant_perts = 0usize;
    let k_restart = (instance.num_operations() / 15).max(2);

    while counter.remaining() > 0 {
        let base_seqs = if stagnant_perts >= ILS_STAGNATION_PERTS {
            counters.restarts_triggered += 1;
            stagnant_perts = 0;
            best_seqs.as_slice()
        } else {
            curr_seqs.as_slice()
        };

        evaluator.restore_machine_seqs(base_seqs);
        let state =
            evaluator
                .analyze_current()?
                .ok_or_else(|| JsspError::InvalidDisjunctiveGraph {
                    detail: "ILS base state became cyclic".to_owned(),
                })?;
        let obr = evaluator.encode_canonical_obr(&state);
        let mut genes = obr.to_job_indices();

        let num_cycles =
            if counters.restarts_triggered > 0 && base_seqs.as_ptr() == best_seqs.as_ptr() {
                k_restart
            } else {
                1
            };
        for _ in 0..num_cycles {
            perturb_obr_three_cycle(&mut genes, instance.num_jobs(), &mut streams.perturbation)?;
            counters.perturbations_applied += 1;
        }

        if !counter.try_consume_search() {
            return Ok((curr_cmax, SearchTerminationReason::BudgetExhausted));
        }
        let pert_obr = ObrChromosome::from_job_indices(&genes);
        let pert_sched = decoder_ws.decode(instance, &pert_obr, true)?;
        evaluator.load_from_schedule(&pert_sched)?;

        // Audit adapter left-shifts relative to target machine-sequence state if any
        if let Some(ls_check) = evaluator.analyze_current()? {
            for sched_op in pert_sched.operations() {
                let flat = sched_op
                    .op_id
                    .flat_index(instance.num_jobs(), instance.num_machines())?;
                if sched_op.start_time < ls_check.heads[flat] {
                    counters.adapter_left_shift_count += 1;
                }
            }
        }

        if pert_sched.makespan() < *best_makespan {
            *best_makespan = pert_sched.makespan();
            *best_found_eval = counter.total();
            *best_seqs = evaluator.snapshot_machine_seqs();
            best_milestones.push((*best_found_eval, pert_sched.makespan()));
            stagnant_perts = 0;
        }

        let prev_best = *best_makespan;
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

        if *best_makespan < prev_best {
            stagnant_perts = 0;
        } else {
            stagnant_perts += 1;
        }

        if ls_cmax <= curr_cmax {
            curr_cmax = ls_cmax;
            curr_seqs = evaluator.snapshot_machine_seqs();
            counters.perturbations_accepted += 1;
        }
    }

    Ok((curr_cmax, SearchTerminationReason::BudgetExhausted))
}
