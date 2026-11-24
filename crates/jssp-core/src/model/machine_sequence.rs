//! Ordered sequence of operations processed on a single machine.

use crate::model::identifiers::{MachineId, OperationId};

/// Ordered sequence of operations assigned to a given machine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MachineSequence {
    /// Target machine identifier.
    pub machine: MachineId,
    /// Operations in processing order on `machine`.
    pub operations: Vec<OperationId>,
}

impl MachineSequence {
    /// Constructs a `MachineSequence` for `machine` with ordered `operations`.
    #[must_use]
    pub fn new(machine: MachineId, operations: Vec<OperationId>) -> Self {
        Self {
            machine,
            operations,
        }
    }
}
