//! Independent intra-job technological precedence validator.

use crate::error::JsspError;
use crate::model::{JsspInstance, OperationId, Schedule, ScheduledOperation};

/// Verifies that for every job `j` and consecutive operation pair `(k, k + 1)`,
/// `C_{j,k} <= S_{j,k+1}` holds.
///
/// # Errors
/// Returns [`JsspError::EmptySchedule`], [`JsspError::UnexpectedOperation`],
/// [`JsspError::DuplicateOperation`], [`JsspError::MissingOperation`], or
/// [`JsspError::JobPrecedenceViolation`].
pub fn check_job_precedence(instance: &JsspInstance, schedule: &Schedule) -> Result<(), JsspError> {
    if schedule.operations().is_empty() {
        return Err(JsspError::EmptySchedule);
    }

    let num_jobs = instance.num_jobs();
    let num_machines = instance.num_machines();
    let mut table: Vec<Option<ScheduledOperation>> = vec![None; instance.num_operations()];

    for &sched_op in schedule.operations() {
        let flat = sched_op.op_id.flat_index(num_jobs, num_machines)?;
        if table[flat].is_some() {
            return Err(JsspError::DuplicateOperation {
                job: sched_op.op_id.job.0,
                op_index: sched_op.op_id.op_index.0,
            });
        }
        table[flat] = Some(sched_op);
    }

    for job in 0..num_jobs {
        let first_flat = OperationId::from_indices(job, 0).flat_index(num_jobs, num_machines)?;
        let mut prev = table[first_flat].ok_or(JsspError::MissingOperation { job, op_index: 0 })?;

        for k in 1..num_machines {
            let next_flat = OperationId::from_indices(job, k).flat_index(num_jobs, num_machines)?;
            let next = table[next_flat].ok_or(JsspError::MissingOperation { job, op_index: k })?;

            if prev.completion_time > next.start_time {
                return Err(JsspError::JobPrecedenceViolation {
                    job,
                    prev_op_index: k - 1,
                    next_op_index: k,
                    prev_completion: prev.completion_time,
                    next_start: next.start_time,
                });
            }
            prev = next;
        }
    }

    Ok(())
}
