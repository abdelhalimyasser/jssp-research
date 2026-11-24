//! Pure integer makespan (`C_max`) objective evaluation independent of the decoder.

use crate::error::JsspError;
use crate::model::{JsspInstance, OperationId, Schedule};

/// Evaluates makespan `C_max = max_{(j,k)} C_{j,k}` directly from scheduled operations,
/// verifying complete non-duplicate coverage of all instance operations.
///
/// # Errors
/// Returns [`JsspError::EmptySchedule`], [`JsspError::UnexpectedOperation`],
/// [`JsspError::DuplicateOperation`], or [`JsspError::MissingOperation`] if the schedule
/// does not cover every operation of `instance` exactly once.
pub fn evaluate_makespan(instance: &JsspInstance, schedule: &Schedule) -> Result<u64, JsspError> {
    if schedule.operations().is_empty() {
        return Err(JsspError::EmptySchedule);
    }

    let num_jobs = instance.num_jobs();
    let num_machines = instance.num_machines();
    let mut seen = vec![false; instance.num_operations()];
    let mut max_completion: u64 = 0;

    for sched_op in schedule.operations() {
        let flat = sched_op.op_id.flat_index(num_jobs, num_machines)?;
        if seen[flat] {
            return Err(JsspError::DuplicateOperation {
                job: sched_op.op_id.job.0,
                op_index: sched_op.op_id.op_index.0,
            });
        }
        seen[flat] = true;
        max_completion = max_completion.max(sched_op.completion_time);
    }

    for (flat, was_seen) in seen.into_iter().enumerate() {
        if !was_seen {
            let op_id = OperationId::from_flat_index(flat, num_jobs, num_machines)?;
            return Err(JsspError::MissingOperation {
                job: op_id.job.0,
                op_index: op_id.op_index.0,
            });
        }
    }

    Ok(max_completion)
}
