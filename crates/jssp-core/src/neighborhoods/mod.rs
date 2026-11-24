//! Disjunctive-graph critical-block neighborhoods `N5`–`N8` (`D03-A`).

pub mod moves;
pub mod n5;
pub mod n6;
pub mod n7;
pub mod n8;
pub mod validation;

pub use moves::{GraphMove, MoveKind, NeighborhoodId, apply_move_to_slice, undo_move_on_slice};
pub use n5::generate_n5;
pub use n6::generate_n6;
pub use n7::generate_n7;
pub use n8::generate_n8;
pub use validation::GeneratedNeighborhood;

use crate::local_search::evaluation::{EvaluatedGraphState, FastGraphEvaluator};

/// Dispatches canonical neighborhood move enumeration for `id`.
#[must_use]
pub fn generate_neighborhood(
    id: NeighborhoodId,
    eval: &FastGraphEvaluator,
    state: &EvaluatedGraphState,
) -> GeneratedNeighborhood {
    match id {
        NeighborhoodId::N5 => generate_n5(eval, state),
        NeighborhoodId::N6 => generate_n6(eval, state),
        NeighborhoodId::N7 => generate_n7(eval, state),
        NeighborhoodId::N8 => generate_n8(eval, state),
    }
}
