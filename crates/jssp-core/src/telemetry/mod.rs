//! Execution counters, Layer A population diagnostics, and Layer B GA telemetry.

pub mod counters;
pub mod run_metrics;

pub use counters::EvalCounter;
pub use run_metrics::{
    GenerationSnapshot, PhaseARunResult, PopulationDiagnostics, compute_1d_ks_discrepancy,
    compute_anytime_cmax_auc, compute_obr_hamming_diversity, compute_schedule_diversity,
    count_unique_chromosomes, schedule_machine_sequence_hash, summarize_population_diagnostics,
};
