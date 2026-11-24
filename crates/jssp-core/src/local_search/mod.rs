//! Exact disjunctive-graph evaluation and Phase B descent algorithms (`B01`–`B12`).

pub mod descent;
pub mod evaluation;

pub use descent::{
    PhaseBRunResult, SearchCounters, SearchTerminationReason, compute_stepwise_anytime_auc,
    run_first_improvement, run_random_improvement, run_steepest_descent,
};
pub use evaluation::{CriticalBlockSlice, EvaluatedGraphState, FastGraphEvaluator};
