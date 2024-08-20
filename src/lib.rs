pub mod igwas_prod;

use igwas::{run_cli, InputArguments};
use mdav::mdav;
use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::{exceptions::PyRuntimeError, prelude::*};

use igwas_prod::{run_igwas, Projection};

/// Module for low-level functionality in Rust.
#[pymodule]
fn _lowlevel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(mdav_impl, m)?)?;
    m.add_function(wrap_pyfunction!(igwas_impl, m)?)?;
    m.add_class::<Projection>()?;
    m.add_function(wrap_pyfunction!(run_igwas, m)?)?;
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

#[pyfunction]
fn igwas_impl(
    projection_matrix: String,
    covariance_matrix: String,
    gwas_results: Vec<String>,
    output_file: String,
    num_covar: usize,
    chunksize: usize,
    variant_id: String,
    beta: String,
    std_error: String,
    sample_size: String,
    num_threads: usize,
    capacity: usize,
    compress: bool,
    quiet: bool,
) -> PyResult<()> {
    let args = InputArguments {
        projection_matrix,
        covariance_matrix,
        gwas_results,
        output_file,
        num_covar,
        chunksize,
        variant_id,
        beta,
        std_error,
        sample_size,
        num_threads,
        capacity,
        compress,
        quiet,
        write_phenotype_id: false,
    };
    match run_cli(args) {
        Ok(_) => Ok(()),
        Err(e) => Err(PyRuntimeError::new_err(format!("{e}"))),
    }
}
