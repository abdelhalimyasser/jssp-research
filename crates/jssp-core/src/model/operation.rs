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
    /// Non-negative integer processing duration (`>= 0`, with instantaneous `0` supported).
    pub processing_time: u64,
}

impl Operation {
    /// Constructs a validated `Operation` with non-negative processing time (`duration >= 0`).
    ///
    /// Note: [`crate::model::JsspInstance::new`] enforces that every job has strictly positive
    /// total processing time across its operations.
    pub fn new(
        id: OperationId,
        machine: MachineId,
        processing_time: u64,
    ) -> Result<Self, JsspError> {
        Ok(Self {
            id,
            machine,
            processing_time,
        })
    }
}
