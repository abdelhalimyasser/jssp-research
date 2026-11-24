//! Specification of a single classical JSSP operation.

use crate::error::JsspError;
use crate::model::identifiers::{MachineId, OperationId};

/// Immutable specification of one operation in a classical JSSP instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Operation {
    /// Canonical operation identifier `(job, op_index)`.
    pub id: OperationId,
    /// Fixed required machine.
    pub machine: MachineId,
    /// Strictly positive integer processing duration.
    pub processing_time: u64,
}

impl Operation {
    /// Constructs a validated `Operation` with positive processing time (`duration > 0`).
    ///
    /// # Errors
    /// Returns [`JsspError::NonPositiveProcessingTime`] if `processing_time == 0`.
    pub fn new(
        id: OperationId,
        machine: MachineId,
        processing_time: u64,
    ) -> Result<Self, JsspError> {
        if processing_time == 0 {
            return Err(JsspError::NonPositiveProcessingTime {
                job: id.job.0,
                op_index: id.op_index.0,
                duration: processing_time,
            });
        }
        Ok(Self {
            id,
            machine,
            processing_time,
        })
    }
}
