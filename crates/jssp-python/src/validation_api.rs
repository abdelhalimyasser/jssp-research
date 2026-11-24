//! PyO3 OBR, schedule feasibility, and makespan validation entrypoints.

use jssp_core::feasibility::validate_schedule as core_validate_schedule;
use jssp_core::objective::evaluate_makespan as core_evaluate_makespan;
use jssp_core::representation::ObrChromosome;
use pyo3::prelude::*;

use crate::instance_api::{jssp_err_to_py, parse_instance};
use crate::result_api::{ScheduledOpTuple, py_ops_to_schedule};

/// Validates an OBR chromosome against a JSSP instance, returning `True` or raising `ValueError`.
#[pyfunction]
pub fn validate_obr(jobs: Vec<Vec<(usize, u64)>>, chromosome: Vec<usize>) -> PyResult<bool> {
    let instance = parse_instance(jobs)?;
    let obr = ObrChromosome::from_job_indices(&chromosome);
    obr.validate(&instance).map_err(jssp_err_to_py)?;
    Ok(true)
}

/// Independently validates a full schedule against all classical JSSP feasibility constraints.
#[pyfunction]
pub fn validate_schedule(
    jobs: Vec<Vec<(usize, u64)>>,
    operations: Vec<ScheduledOpTuple>,
    makespan: u64,
) -> PyResult<bool> {
    let instance = parse_instance(jobs)?;
    let schedule = py_ops_to_schedule(
        instance.num_jobs(),
        instance.num_machines(),
        operations,
        makespan,
    );
    core_validate_schedule(&instance, &schedule).map_err(jssp_err_to_py)?;
    Ok(true)
}

/// Independently recomputes makespan `C_max` from scheduled operation completion times.
#[pyfunction]
pub fn evaluate_makespan(
    jobs: Vec<Vec<(usize, u64)>>,
    operations: Vec<ScheduledOpTuple>,
    makespan: u64,
) -> PyResult<u64> {
    let instance = parse_instance(jobs)?;
    let schedule = py_ops_to_schedule(
        instance.num_jobs(),
        instance.num_machines(),
        operations,
        makespan,
    );
    core_evaluate_makespan(&instance, &schedule).map_err(jssp_err_to_py)
}
