//! Genetic operators and Classical GA (`C01`) engine (`D07-A`).

pub mod crossover;
pub mod elitism;
pub mod mutation;
pub mod population;
pub mod selection;

pub use crossover::pox_crossover;
pub use elitism::ranked_population_indices;
pub use mutation::mutate_obr;
pub use population::{
    ClassicalGaConfig, PhaseABatchTask, default_phase_a_ga_budget, evaluate_initializer_only,
    run_phase_a_batch, run_phase_a_ga,
};
pub use selection::tournament_select;
