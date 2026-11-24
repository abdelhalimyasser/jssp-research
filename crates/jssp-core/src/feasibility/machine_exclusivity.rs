//! Independent machine exclusivity (non-overlap) validator.

use crate::error::JsspError;
use crate::model::{JsspInstance, Schedule, ScheduledOperation};

/// Verifies that no two distinct operations assigned to the same machine overlap
/// in `[S_a, C_a)` and `[S_b, C_b)` (i.e., `C_a <= S_b || C_b <= S_a`).
///
/// # Errors
/// Returns [`JsspError::EmptySchedule`], [`JsspError::InvalidMachineId`], or
/// [`JsspError::MachineOverlapViolation`].
pub fn check_machine_exclusivity(
    instance: &JsspInstance,
    schedule: &Schedule,
) -> Result<(), JsspError> {
    if schedule.operations().is_empty() {
        return Err(JsspError::EmptySchedule);
    }

    let num_machines = instance.num_machines();
    let mut by_machine: Vec<Vec<ScheduledOperation>> = vec![Vec::new(); num_machines];

    for &sched_op in schedule.operations() {
        let machine_idx = sched_op.machine.0;
        if machine_idx >= num_machines {
            return Err(JsspError::InvalidMachineId {
                job: sched_op.op_id.job.0,
                op_index: sched_op.op_id.op_index.0,
                machine: machine_idx,
                num_machines,
            });
        }
        by_machine[machine_idx].push(sched_op);
    }

    for (machine_idx, mut ops) in by_machine.into_iter().enumerate() {
        ops.sort_by_key(|op| {
            (
                op.start_time,
                op.completion_time,
                op.op_id.job,
                op.op_id.op_index,
            )
        });

        for i in 0..ops.len() {
            let a = ops[i];
            for &b in ops.iter().skip(i + 1) {
                let disjoint =
                    a.completion_time <= b.start_time || b.completion_time <= a.start_time;
                if !disjoint {
                    return Err(JsspError::MachineOverlapViolation {
                        machine: machine_idx,
                        op_a: (a.op_id.job.0, a.op_id.op_index.0),
                        interval_a: (a.start_time, a.completion_time),
                        op_b: (b.op_id.job.0, b.op_id.op_index.0),
                        interval_b: (b.start_time, b.completion_time),
                    });
                }

                // Because ops are sorted by non-decreasing start_time, any subsequent valid
                // interval with S_k >= S_b >= C_a cannot overlap [S_a, C_a).
                if a.completion_time <= b.start_time && b.start_time <= b.completion_time {
                    break;
                }
            }
        }
    }

    Ok(())
}
