//! Strongly typed 0-based identifiers for jobs, machines, and operations.

use std::fmt;

use crate::error::JsspError;

/// Zero-based job identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct JobId(pub usize);

impl JobId {
    /// Constructs a `JobId` from a zero-based index.
    #[must_use]
    pub const fn new(index: usize) -> Self {
        Self(index)
    }

    /// Returns the underlying zero-based `usize` index.
    #[must_use]
    pub const fn index(self) -> usize {
        self.0
    }
}

impl From<usize> for JobId {
    fn from(value: usize) -> Self {
        Self(value)
    }
}

impl fmt::Display for JobId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "J{}", self.0)
    }
}

/// Zero-based machine identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MachineId(pub usize);

impl MachineId {
    /// Constructs a `MachineId` from a zero-based index.
    #[must_use]
    pub const fn new(index: usize) -> Self {
        Self(index)
    }

    /// Returns the underlying zero-based `usize` index.
    #[must_use]
    pub const fn index(self) -> usize {
        self.0
    }
}

impl From<usize> for MachineId {
    fn from(value: usize) -> Self {
        Self(value)
    }
}

impl fmt::Display for MachineId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "M{}", self.0)
    }
}

/// Zero-based operation index within a job's fixed technological sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OperationIndex(pub usize);

impl OperationIndex {
    /// Constructs an `OperationIndex` from a zero-based position within a job.
    #[must_use]
    pub const fn new(index: usize) -> Self {
        Self(index)
    }

    /// Returns the underlying zero-based `usize` index.
    #[must_use]
    pub const fn index(self) -> usize {
        self.0
    }
}

impl From<usize> for OperationIndex {
    fn from(value: usize) -> Self {
        Self(value)
    }
}

impl fmt::Display for OperationIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "k{}", self.0)
    }
}

/// Canonical identifier for an operation `(job, op_index)` within a rectangular JSSP instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OperationId {
    /// Owning job identifier.
    pub job: JobId,
    /// Position of the operation within its job's fixed routing.
    pub op_index: OperationIndex,
}

impl OperationId {
    /// Constructs an `OperationId` from typed `JobId` and `OperationIndex`.
    #[must_use]
    pub const fn new(job: JobId, op_index: OperationIndex) -> Self {
        Self { job, op_index }
    }

    /// Constructs an `OperationId` directly from zero-based indices.
    #[must_use]
    pub const fn from_indices(job: usize, op_index: usize) -> Self {
        Self {
            job: JobId(job),
            op_index: OperationIndex(op_index),
        }
    }

    /// Maps `(job, op_index)` to row-major flat index `job * num_machines + op_index`.
    ///
    /// # Errors
    /// Returns [`JsspError::EmptyJobs`], [`JsspError::EmptyMachines`],
    /// [`JsspError::UnexpectedOperation`], or [`JsspError::DimensionOverflow`]
    /// if dimensions or bounds are invalid.
    pub fn flat_index(&self, num_jobs: usize, num_machines: usize) -> Result<usize, JsspError> {
        if num_jobs == 0 {
            return Err(JsspError::EmptyJobs);
        }
        if num_machines == 0 {
            return Err(JsspError::EmptyMachines);
        }
        if self.job.0 >= num_jobs || self.op_index.0 >= num_machines {
            return Err(JsspError::UnexpectedOperation {
                job: self.job.0,
                op_index: self.op_index.0,
                num_jobs,
                num_machines,
            });
        }
        let row_offset = self
            .job
            .0
            .checked_mul(num_machines)
            .ok_or(JsspError::DimensionOverflow)?;
        row_offset
            .checked_add(self.op_index.0)
            .ok_or(JsspError::DimensionOverflow)
    }

    /// Recovers `(job, op_index)` from a row-major flat index in `[0, num_jobs * num_machines)`.
    ///
    /// # Errors
    /// Returns [`JsspError::EmptyJobs`], [`JsspError::EmptyMachines`],
    /// [`JsspError::DimensionOverflow`], or [`JsspError::UnexpectedOperation`]
    /// if the flat index or dimensions are invalid.
    pub fn from_flat_index(
        flat: usize,
        num_jobs: usize,
        num_machines: usize,
    ) -> Result<Self, JsspError> {
        if num_jobs == 0 {
            return Err(JsspError::EmptyJobs);
        }
        if num_machines == 0 {
            return Err(JsspError::EmptyMachines);
        }
        let total = num_jobs
            .checked_mul(num_machines)
            .ok_or(JsspError::DimensionOverflow)?;
        let job = flat / num_machines;
        let op_index = flat % num_machines;
        if flat >= total {
            return Err(JsspError::UnexpectedOperation {
                job,
                op_index,
                num_jobs,
                num_machines,
            });
        }
        Ok(Self::from_indices(job, op_index))
    }
}

impl fmt::Display for OperationId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "O({}, {})", self.job.0, self.op_index.0)
    }
}
