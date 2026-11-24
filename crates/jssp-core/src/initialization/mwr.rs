//! Deterministic Most Work Remaining (MWR) constructive OBR initializer (`A02`).

use crate::error::JsspError;
use crate::initialization::qmc::discrete_mapping::initial_job_remaining_work;
use crate::model::{JobId, JsspInstance};
use crate::representation::ObrChromosome;

/// Generates the deterministic dynamic MWR OBR chromosome for `instance` (`A02`).
///
/// At each construction step, picks the eligible job with maximum remaining processing
/// time `W_j(t)`, breaking all ties deterministically by lowest [`JobId`].
///
/// # Errors
/// Returns [`JsspError`] if instance operation lookups fail.
pub fn deterministic_mwr_chromosome(instance: &JsspInstance) -> Result<ObrChromosome, JsspError> {
    let num_jobs = instance.num_jobs();
    let num_machines = instance.num_machines();
    let total_ops = instance.num_operations();

    let mut rem_ops = vec![num_machines; num_jobs];
    let mut next_op = vec![0usize; num_jobs];
    let mut rem_work = initial_job_remaining_work(instance);
    let mut genes = Vec::with_capacity(total_ops);

    for _ in 0..total_ops {
        let mut best_job: Option<usize> = None;
        let mut best_work: u64 = 0;

        for (job_idx, (&rem, &work)) in rem_ops.iter().zip(rem_work.iter()).enumerate() {
            if rem == 0 {
                continue;
            }
            let is_better = match best_job {
                None => true,
                Some(_) => work > best_work,
            };
            if is_better {
                best_job = Some(job_idx);
                best_work = work;
            }
        }

        let chosen = best_job.ok_or(JsspError::EmptyJobs)?;
        genes.push(JobId(chosen));
        let op = instance.operation_by_indices(chosen, next_op[chosen])?;
        rem_work[chosen] = rem_work[chosen].saturating_sub(op.processing_time);
        next_op[chosen] += 1;
        rem_ops[chosen] -= 1;
    }

    Ok(ObrChromosome::new(genes))
}

/// Replicates the deterministic MWR chromosome `pop_size` times without artificial noise (`A02`).
///
/// # Errors
/// Returns [`JsspError`] if chromosome construction fails.
pub fn generate_deterministic_mwr_population(
    instance: &JsspInstance,
    pop_size: usize,
) -> Result<Vec<ObrChromosome>, JsspError> {
    let base = deterministic_mwr_chromosome(instance)?;
    Ok(vec![base; pop_size])
}
