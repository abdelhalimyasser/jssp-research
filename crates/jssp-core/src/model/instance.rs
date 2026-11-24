//! Rectangular classical Job-Shop Scheduling Problem instance model.

use crate::error::JsspError;
use crate::model::identifiers::{JobId, MachineId, OperationId};
use crate::model::operation::Operation;

/// Validated deterministic classical rectangular JSSP instance.
///
/// Every job `0..num_jobs` consists of `num_machines` operations whose machines
/// form a permutation of `0..num_machines`, each with strictly positive `u64`
/// processing time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JsspInstance {
    num_jobs: usize,
    num_machines: usize,
    num_operations: usize,
    total_processing_time: u64,
    jobs: Vec<Vec<Operation>>,
}

impl JsspInstance {
    /// Constructs and validates a rectangular JSSP instance from per-job `(machine, duration)` lists.
    ///
    /// # Errors
    /// Rejects empty jobs, empty machines, non-rectangular job rows, out-of-range
    /// machine IDs, duplicate machines within a job, zero processing times, or
    /// overflowing dimensions / total processing duration.
    pub fn new(raw_jobs: Vec<Vec<(usize, u64)>>) -> Result<Self, JsspError> {
        let num_jobs = raw_jobs.len();
        if num_jobs == 0 {
            return Err(JsspError::EmptyJobs);
        }

        let num_machines = raw_jobs[0].len();
        if num_machines == 0 {
            return Err(JsspError::EmptyMachines);
        }

        let num_operations = num_jobs
            .checked_mul(num_machines)
            .ok_or(JsspError::DimensionOverflow)?;

        let mut total_processing_time: u64 = 0;
        let mut jobs = Vec::with_capacity(num_jobs);

        for (job_idx, row) in raw_jobs.into_iter().enumerate() {
            if row.len() != num_machines {
                return Err(JsspError::NonRectangularJobLength {
                    job: job_idx,
                    expected: num_machines,
                    actual: row.len(),
                });
            }

            let mut seen_machines = vec![false; num_machines];
            let mut job_ops = Vec::with_capacity(num_machines);

            for (op_idx, (machine, duration)) in row.into_iter().enumerate() {
                if machine >= num_machines {
                    return Err(JsspError::InvalidMachineId {
                        job: job_idx,
                        op_index: op_idx,
                        machine,
                        num_machines,
                    });
                }
                if seen_machines[machine] {
                    return Err(JsspError::DuplicateMachineInJob {
                        job: job_idx,
                        machine,
                    });
                }
                seen_machines[machine] = true;

                total_processing_time = total_processing_time
                    .checked_add(duration)
                    .ok_or(JsspError::TimeOverflow)?;

                let op = Operation::new(
                    OperationId::from_indices(job_idx, op_idx),
                    MachineId(machine),
                    duration,
                )?;
                job_ops.push(op);
            }

            jobs.push(job_ops);
        }

        Ok(Self {
            num_jobs,
            num_machines,
            num_operations,
            total_processing_time,
            jobs,
        })
    }

    /// Constructs a validated `JsspInstance` from parallel machine and processing-time matrices.
    ///
    /// # Errors
    /// Returns [`JsspError`] if matrices are empty, have mismatched dimensions,
    /// are non-rectangular, or violate classical JSSP constraints.
    pub fn from_matrices(
        machines: &[Vec<usize>],
        processing_times: &[Vec<u64>],
    ) -> Result<Self, JsspError> {
        let num_jobs = machines.len();
        if num_jobs == 0 {
            return Err(JsspError::EmptyJobs);
        }
        if processing_times.len() != num_jobs {
            return Err(JsspError::NonRectangularJobLength {
                job: processing_times.len().min(num_jobs),
                expected: num_jobs,
                actual: processing_times.len(),
            });
        }

        let num_machines = machines[0].len();
        if num_machines == 0 {
            return Err(JsspError::EmptyMachines);
        }

        let mut raw_jobs = Vec::with_capacity(num_jobs);
        for (job_idx, (m_row, p_row)) in machines.iter().zip(processing_times.iter()).enumerate() {
            if m_row.len() != num_machines {
                return Err(JsspError::NonRectangularJobLength {
                    job: job_idx,
                    expected: num_machines,
                    actual: m_row.len(),
                });
            }
            if p_row.len() != num_machines {
                return Err(JsspError::NonRectangularJobLength {
                    job: job_idx,
                    expected: num_machines,
                    actual: p_row.len(),
                });
            }
            let combined = m_row.iter().copied().zip(p_row.iter().copied()).collect();
            raw_jobs.push(combined);
        }

        Self::new(raw_jobs)
    }

    /// Number of jobs `J`.
    #[must_use]
    pub const fn num_jobs(&self) -> usize {
        self.num_jobs
    }

    /// Number of machines `M` (also the number of operations per job).
    #[must_use]
    pub const fn num_machines(&self) -> usize {
        self.num_machines
    }

    /// Total number of operations `L = J * M`.
    #[must_use]
    pub const fn num_operations(&self) -> usize {
        self.num_operations
    }

    /// Sum of processing times across all `J * M` operations.
    #[must_use]
    pub const fn total_processing_time(&self) -> u64 {
        self.total_processing_time
    }

    /// Returns all jobs and their fixed operation sequences.
    #[must_use]
    pub fn jobs(&self) -> &[Vec<Operation>] {
        &self.jobs
    }

    /// Returns the ordered slice of operations for `job`, or `None` if out of bounds.
    #[must_use]
    pub fn job_operations(&self, job: JobId) -> Option<&[Operation]> {
        self.jobs.get(job.0).map(Vec::as_slice)
    }

    /// Returns the `Operation` identified by `op_id`, or `None` if out of bounds.
    #[must_use]
    pub fn operation(&self, op_id: OperationId) -> Option<&Operation> {
        self.jobs
            .get(op_id.job.0)
            .and_then(|ops| ops.get(op_id.op_index.0))
    }

    /// Looks up an operation by raw indices `(job, op_index)` with explicit error context.
    ///
    /// # Errors
    /// Returns [`JsspError::UnexpectedOperation`] if indices are outside instance bounds.
    pub fn operation_by_indices(
        &self,
        job: usize,
        op_index: usize,
    ) -> Result<&Operation, JsspError> {
        self.jobs
            .get(job)
            .and_then(|ops| ops.get(op_index))
            .ok_or(JsspError::UnexpectedOperation {
                job,
                op_index,
                num_jobs: self.num_jobs,
                num_machines: self.num_machines,
            })
    }
}
