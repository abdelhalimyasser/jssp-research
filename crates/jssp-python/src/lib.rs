//! PyO3 native extension module (`jssp_research._native`).
//!
//! Exposes coarse-grained classical JSSP instance validation, OBR decoding,
//! independent schedule feasibility verification, and makespan evaluation
//! backed by `jssp-core`.

mod graph_api;
mod instance_api;
mod phase_a_api;
mod result_api;
mod run_api;
mod validation_api;

use pyo3::prelude::*;

/// Returns a diagnostic identification string confirming the active Rust core bridge.
#[pyfunction]
fn bridge_info() -> &'static str {
    jssp_core::core_ident()
}

/// Returns the compiled `jssp-core` semantic version string.
#[pyfunction]
fn core_version() -> &'static str {
    jssp_core::crate_version()
}

/// Native PyO3 module exported as `jssp_research._native`.
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", jssp_core::crate_version())?;
    m.add_function(wrap_pyfunction!(bridge_info, m)?)?;
    m.add_function(wrap_pyfunction!(core_version, m)?)?;
    m.add_function(wrap_pyfunction!(instance_api::validate_instance, m)?)?;
    m.add_function(wrap_pyfunction!(validation_api::validate_obr, m)?)?;
    m.add_function(wrap_pyfunction!(run_api::resolve_obr_operations, m)?)?;
    m.add_function(wrap_pyfunction!(run_api::decode_obr, m)?)?;
    m.add_function(wrap_pyfunction!(run_api::decode_obr_append_only, m)?)?;
    m.add_function(wrap_pyfunction!(validation_api::validate_schedule, m)?)?;
    m.add_function(wrap_pyfunction!(validation_api::evaluate_makespan, m)?)?;
    m.add_function(wrap_pyfunction!(run_api::frozen_master_seeds, m)?)?;
    m.add_function(wrap_pyfunction!(run_api::rng_metadata, m)?)?;
    m.add_function(wrap_pyfunction!(run_api::derive_rng_streams, m)?)?;
    m.add_function(wrap_pyfunction!(run_api::derive_rng_streams_with_seed, m)?)?;
    m.add_function(wrap_pyfunction!(run_api::rng_sample_range_and_shuffle, m)?)?;
    m.add_function(wrap_pyfunction!(graph_api::analyze_schedule_graph, m)?)?;
    m.add_function(wrap_pyfunction!(graph_api::analyze_obr_graph, m)?)?;
    m.add_function(wrap_pyfunction!(
        graph_api::validate_machine_sequences_dag,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        graph_api::build_machine_sequence_graph,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        graph_api::encode_machine_sequences_to_obr,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        graph_api::round_trip_machine_sequences,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(graph_api::round_trip_obr_schedule, m)?)?;
    m.add_function(wrap_pyfunction!(phase_a_api::list_phase_a_initializers, m)?)?;
    m.add_function(wrap_pyfunction!(phase_a_api::default_phase_a_ga_budget, m)?)?;
    m.add_function(wrap_pyfunction!(
        phase_a_api::evaluate_phase_a_initializer,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(phase_a_api::run_phase_a_single, m)?)?;
    m.add_function(wrap_pyfunction!(phase_a_api::run_phase_a_batch, m)?)?;
    Ok(())
}
