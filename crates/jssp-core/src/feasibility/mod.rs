//! Independent schedule feasibility verification for classical JSSP.

pub mod disjunctive_graph;
pub mod machine_exclusivity;
pub mod precedence;

use crate::error::JsspError;
use crate::model::{JsspInstance, OperationId, Schedule, ScheduledOperation};
use crate::objective::evaluate_makespan;

pub use machine_exclusivity::check_machine_exclusivity;
pub use precedence::check_job_precedence;

/// Independently validates every classical JSSP feasibility invariant on `schedule`:
/// 1. Non-empty operation list and exact operation coverage (`0..J` x `0..M`, no duplicates or omissions).
/// 2. Instance metadata consistency (matching machine assignment and processing duration).
/// 3. Interval arithmetic consistency (`C_{j,k} == S_{j,k}.checked_add(p_{j,k})`).
/// 4. Intra-job precedence (`C_{j,k} <= S_{j,k+1}`).
/// 5. Machine exclusivity (`[S_a, C_a)` and `[S_b, C_b)` disjoint on every machine).
/// 6. Makespan consistency (`schedule.makespan() == max_{(j,k)} C_{j,k}`).
///
/// # Errors
/// Returns a typed [`JsspError`] describing the first violated constraint.
pub fn validate_schedule(instance: &JsspInstance, schedule: &Schedule) -> Result<(), JsspError> {
    if schedule.operations().is_empty() {
        return Err(JsspError::EmptySchedule);
    }

    let num_jobs = instance.num_jobs();
    let num_machines = instance.num_machines();
    let mut seen: Vec<Option<ScheduledOperation>> = vec![None; instance.num_operations()];

    for &sched_op in schedule.operations() {
        let flat = sched_op.op_id.flat_index(num_jobs, num_machines)?;
        if seen[flat].is_some() {
            return Err(JsspError::DuplicateOperation {
                job: sched_op.op_id.job.0,
                op_index: sched_op.op_id.op_index.0,
            });
        }
        seen[flat] = Some(sched_op);
    }

    for (flat, entry) in seen.into_iter().enumerate() {
        let op_id = OperationId::from_flat_index(flat, num_jobs, num_machines)?;
        let sched_op = entry.ok_or(JsspError::MissingOperation {
            job: op_id.job.0,
            op_index: op_id.op_index.0,
        })?;
        let spec_op = instance.operation_by_indices(op_id.job.0, op_id.op_index.0)?;

        if sched_op.machine != spec_op.machine {
            return Err(JsspError::OperationMachineMismatch {
                job: op_id.job.0,
                op_index: op_id.op_index.0,
                expected_machine: spec_op.machine.0,
                actual_machine: sched_op.machine.0,
            });
        }
        if sched_op.processing_time != spec_op.processing_time {
            return Err(JsspError::OperationProcessingTimeMismatch {
                job: op_id.job.0,
                op_index: op_id.op_index.0,
                expected_duration: spec_op.processing_time,
                actual_duration: sched_op.processing_time,
            });
        }

        let expected_end = sched_op
            .start_time
            .checked_add(spec_op.processing_time)
            .ok_or(JsspError::TimeOverflow)?;
        if sched_op.completion_time != expected_end {
            return Err(JsspError::ProcessingDurationMismatch {
                job: op_id.job.0,
                op_index: op_id.op_index.0,
                start_time: sched_op.start_time,
                completion_time: sched_op.completion_time,
                expected_duration: spec_op.processing_time,
            });
        }
    }

    check_job_precedence(instance, schedule)?;
    check_machine_exclusivity(instance, schedule)?;

    let recomputed = evaluate_makespan(instance, schedule)?;
    if schedule.makespan() != recomputed {
        return Err(JsspError::MakespanMismatch {
            reported: schedule.makespan(),
            recomputed,
        });
    }

    Ok(())
}
