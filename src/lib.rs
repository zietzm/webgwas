use mdav::mdav;
use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

/// Module for low-level functionality in Rust.
#[pymodule]
fn _lowlevel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(mdav_impl, m)?)?;
    Ok(())
}

/// Anonymize a set of records using the MDAV algorithm.
///
/// # Arguments
///
/// * `records` - A 2D array of records, where each row is a record and each column is a feature.
/// * `k` - The number of features to use for the MDAV algorithm.
///
/// # Returns
///
/// A 2D array of records, where each row is a record and each column is a feature.
#[pyfunction]
fn mdav_impl<'py>(
    py: Python<'py>,
    records: PyReadonlyArray2<'py, f64>,
    k: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let records_vec: Vec<Vec<f64>> = records
        .as_array()
        .outer_iter()
        .map(|row| row.to_vec())
        .collect();

    let result = mdav::mdav(records_vec, k);

    Ok(PyArray2::from_vec2_bound(py, &result)?)
}
