//! Decoded schedule representation and deterministic query helpers.

use crate::model::identifiers::{JobId, MachineId, OperationId};
use crate::model::machine_sequence::MachineSequence;

/// One scheduled operation with concrete integer start and completion times.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScheduledOperation {
    /// Canonical operation identifier `(job, op_index)`.
    pub op_id: OperationId,
    /// Assigned machine identifier.
    pub machine: MachineId,
    /// Non-negative start time `S_{j,k}`.
    pub start_time: u64,
    /// Completion time `C_{j,k} = S_{j,k} + p_{j,k}`.
    pub completion_time: u64,
    /// Operation processing duration `p_{j,k}`.
    pub processing_time: u64,
}

/// Complete schedule for a classical JSSP instance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Schedule {
    num_jobs: usize,
    num_machines: usize,
    operations: Vec<ScheduledOperation>,
    makespan: u64,
}

impl Schedule {
    /// Constructs a `Schedule` preserving the supplied operation records and reported makespan.
    ///
    /// Construction intentionally does not mutate, filter, or auto-correct invalid records
    /// so that [`crate::feasibility::validate_schedule`] can inspect untrusted schedules.
    #[must_use]
    pub fn new(
        num_jobs: usize,
        num_machines: usize,
        operations: Vec<ScheduledOperation>,
        makespan: u64,
    ) -> Self {
        Self {
            num_jobs,
            num_machines,
            operations,
            makespan,
        }
    }

    /// Number of jobs expected by this schedule.
    #[must_use]
    pub const fn num_jobs(&self) -> usize {
        self.num_jobs
    }

    /// Number of machines expected by this schedule.
    #[must_use]
    pub const fn num_machines(&self) -> usize {
        self.num_machines
    }

    /// Reported makespan `C_max`.
    #[must_use]
    pub const fn makespan(&self) -> u64 {
        self.makespan
    }

    /// All scheduled operation records stored in this schedule.
    #[must_use]
    pub fn operations(&self) -> &[ScheduledOperation] {
        &self.operations
    }

    /// Looks up the first scheduled record matching `op_id`.
    #[must_use]
    pub fn operation(&self, op_id: OperationId) -> Option<&ScheduledOperation> {
        self.operations.iter().find(|op| op.op_id == op_id)
    }

    /// Returns scheduled operations belonging to `job`, deterministically sorted by
    /// `(op_index, start_time, completion_time)`.
    #[must_use]
    pub fn job_operations(&self, job: JobId) -> Vec<ScheduledOperation> {
        let mut ops: Vec<ScheduledOperation> = self
            .operations
            .iter()
            .copied()
            .filter(|op| op.op_id.job == job)
            .collect();
        ops.sort_by_key(|op| (op.op_id.op_index, op.start_time, op.completion_time));
        ops
    }

    /// Returns scheduled operations executed on `machine`, deterministically sorted by
    /// `(start_time, completion_time, job, op_index)`.
    #[must_use]
    pub fn machine_operations(&self, machine: MachineId) -> Vec<ScheduledOperation> {
        let mut ops: Vec<ScheduledOperation> = self
            .operations
            .iter()
            .copied()
            .filter(|op| op.machine == machine)
            .collect();
        ops.sort_by_key(|op| {
            (
                op.start_time,
                op.completion_time,
                op.op_id.job,
                op.op_id.op_index,
            )
        });
        ops
    }

    /// Extracts deterministic per-machine operation sequences for `0..num_machines`.
    #[must_use]
    pub fn machine_sequences(&self) -> Vec<MachineSequence> {
        (0..self.num_machines)
            .map(|m| {
                let machine = MachineId(m);
                let ops = self
                    .machine_operations(machine)
                    .into_iter()
                    .map(|op| op.op_id)
                    .collect();
                MachineSequence::new(machine, ops)
            })
            .collect()
    }

    /// Extracts deterministic per-job scheduled operation sequences for `0..num_jobs`.
    #[must_use]
    pub fn job_sequences(&self) -> Vec<Vec<ScheduledOperation>> {
        (0..self.num_jobs)
            .map(|j| self.job_operations(JobId(j)))
            .collect()
    }
}
