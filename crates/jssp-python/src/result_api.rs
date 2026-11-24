//! PyO3 result serialization and schedule deserialization helpers.

use jssp_core::model::{
    JobId, MachineId, OperationId, OperationIndex, Schedule, ScheduledOperation,
};

/// Python-facing flat scheduled operation record:
/// `(job, op_index, machine, start_time, completion_time, processing_time)`.
pub type ScheduledOpTuple = (usize, usize, usize, u64, u64, u64);

/// Python-facing per-machine sequences of `(job, op_index)` pairs.
pub type MachineSequenceList = Vec<Vec<(usize, usize)>>;

/// Python-facing decoded schedule payload: `(makespan, operations, machine_sequences)`.
pub type DecodedSchedulePayload = (u64, Vec<ScheduledOpTuple>, MachineSequenceList);

/// Converts a core Rust [`Schedule`] into Python primitive tuples.
pub(crate) fn schedule_to_py_payload(schedule: &Schedule) -> DecodedSchedulePayload {
    let operations = schedule
        .operations()
        .iter()
        .map(|op| {
            (
                op.op_id.job.0,
                op.op_id.op_index.0,
                op.machine.0,
                op.start_time,
                op.completion_time,
                op.processing_time,
            )
        })
        .collect();

    let machine_sequences = schedule
        .machine_sequences()
        .into_iter()
        .map(|seq| {
            seq.operations
                .into_iter()
                .map(|op_id| (op_id.job.0, op_id.op_index.0))
                .collect()
        })
        .collect();

    (schedule.makespan(), operations, machine_sequences)
}

/// Reconstructs a core [`Schedule`] from Python primitive tuples without filtering
/// so that independent feasibility validators can reject malformed inputs.
pub(crate) fn py_ops_to_schedule(
    num_jobs: usize,
    num_machines: usize,
    operations: Vec<ScheduledOpTuple>,
    makespan: u64,
) -> Schedule {
    let scheduled_ops = operations
        .into_iter()
        .map(
            |(job, op_index, machine, start_time, completion_time, processing_time)| {
                ScheduledOperation {
                    op_id: OperationId::new(JobId(job), OperationIndex(op_index)),
                    machine: MachineId(machine),
                    start_time,
                    completion_time,
                    processing_time,
                }
            },
        )
        .collect();

    Schedule::new(num_jobs, num_machines, scheduled_ops, makespan)
}
