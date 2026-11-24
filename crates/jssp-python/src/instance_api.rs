//! PyO3 instance construction and error translation helpers.

use jssp_core::error::JsspError;
use jssp_core::model::JsspInstance;
use pyo3::exceptions::{PyOverflowError, PyValueError};
use pyo3::prelude::*;

/// Maps typed Rust core errors cleanly to Python exceptions.
pub(crate) fn jssp_err_to_py(err: JsspError) -> PyErr {
    match err {
        JsspError::TimeOverflow | JsspError::DimensionOverflow => {
            PyOverflowError::new_err(err.to_string())
        }
        other => PyValueError::new_err(other.to_string()),
    }
}

/// Parses and validates a rectangular JSSP instance from nested `(machine, duration)` rows.
pub(crate) fn parse_instance(jobs: Vec<Vec<(usize, u64)>>) -> PyResult<JsspInstance> {
    JsspInstance::new(jobs).map_err(jssp_err_to_py)
}

/// Validates a rectangular JSSP instance and returns `(num_jobs, num_machines, num_operations, total_processing_time)`.
#[pyfunction]
pub fn validate_instance(jobs: Vec<Vec<(usize, u64)>>) -> PyResult<(usize, usize, usize, u64)> {
    let instance = parse_instance(jobs)?;
    Ok((
        instance.num_jobs(),
        instance.num_machines(),
        instance.num_operations(),
        instance.total_processing_time(),
    ))
}
