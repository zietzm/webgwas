use std::{fs::File, io::BufWriter, sync::Arc};

use anyhow::{anyhow, Result};
use arrow::{
    array::{Float32Array, Int32Array, LargeStringArray, StructArray},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use itertools::izip;
use parquet::arrow::arrow_reader::{ParquetRecordBatchReader, ParquetRecordBatchReaderBuilder};
use pyo3::{exceptions::PyValueError, prelude::*};
use zstd::stream::AutoFinishEncoder;

#[pyclass]
pub struct Projection {
    #[pyo3(get, set)]
    pub feature_id: Vec<String>,
    #[pyo3(get, set)]
    pub feature_coefficient: Vec<f32>,
    #[pyo3(get, set)]
    pub n_features: usize,
}

#[pymethods]
impl Projection {
    #[new]
    pub fn new(feature_id: Vec<String>, feature_coefficient: Vec<f32>) -> PyResult<Self> {
        if feature_id.len() != feature_coefficient.len() {
            return Err(PyValueError::new_err(
                "feature_id and feature_coefficient must have the same length.",
            ));
        }
        let n_features = feature_coefficient.iter().filter(|&c| c != &0.0).count();
        let result = Projection {
            feature_id,
            feature_coefficient,
            n_features,
        };
        Ok(result)
    }
}

pub fn indirect_std_error(
    projection_variance: f32,
    indirect_genotype_variance: f32,
    indirect_beta: f32,
    indirect_degrees_of_freedom: i32,
) -> f32 {
    let variance_ratio = projection_variance / indirect_genotype_variance;
    let numerator = variance_ratio - indirect_beta.powi(2);
    (numerator / indirect_degrees_of_freedom as f32).sqrt()
}

#[derive(Clone, Debug)]
pub struct SingleVariantRunningStats {
    pub variant_id: String,
    pub beta: f32,
    pub genotype_variance: f32,
    pub degrees_of_freedom: i32,
}

pub struct RunningStats {
    pub variant_id: Vec<String>,
    pub n_variants: usize,
    pub beta: Vec<f32>,
    pub genotype_variance: Vec<f32>,
    pub degrees_of_freedom: Vec<i32>,
}

impl RunningStats {
    pub fn new(n_variants: usize) -> Self {
        RunningStats {
            n_variants,
            variant_id: vec!["".to_string(); n_variants],
            beta: vec![0.0; n_variants],
            genotype_variance: vec![0.0; n_variants],
            degrees_of_freedom: vec![i32::MAX; n_variants],
        }
    }

    pub fn get_single_variant(&self, i: usize) -> SingleVariantRunningStats {
        SingleVariantRunningStats {
            variant_id: self.variant_id[i].clone(),
            beta: self.beta[i],
            genotype_variance: self.genotype_variance[i],
            degrees_of_freedom: self.degrees_of_freedom[i],
        }
    }
}

#[derive(Clone, Debug)]
pub struct SingleVariantResultStats {
    pub variant_id: String,
    pub beta: f32,
    pub std_error: f32,
    pub t_stat: f32,
    pub neg_log_p_value: f32,
    pub sample_size: i32,
    pub allele_frequency: f32,
}

#[derive(Clone, Debug)]
pub struct ResultStats {
    pub variant_id: Vec<String>,
    pub beta: Vec<f32>,
    pub std_error: Vec<f32>,
    pub t_stat: Vec<f32>,
    pub neg_log_p_value: Vec<f32>,
    pub sample_size: Vec<i32>,
    pub allele_frequency: Vec<f32>,
}

impl ResultStats {
    pub fn get_single_variant(&self, i: usize) -> SingleVariantResultStats {
        SingleVariantResultStats {
            variant_id: self.variant_id[i].clone(),
            beta: self.beta[i],
            std_error: self.std_error[i],
            t_stat: self.t_stat[i],
            neg_log_p_value: self.neg_log_p_value[i],
            sample_size: self.sample_size[i],
            allele_frequency: self.allele_frequency[i],
        }
    }
}

pub struct FeatureStats {
    pub variant_id: LargeStringArray,
    pub beta: Float32Array,
    pub genotype_variance: Float32Array,
    pub degrees_of_freedom: Int32Array,
}

fn get_columns(record_batch: &RecordBatch, feature_id: &str) -> Result<FeatureStats> {
    let feature_column = record_batch
        .column_by_name(feature_id)
        .ok_or_else(|| anyhow!("Feature column not found in record batch."))?
        .as_any()
        .downcast_ref::<StructArray>()
        .ok_or_else(|| anyhow!("Feature column {} is not a StructArray.", feature_id))?;

    let variant_ids = record_batch
        .column_by_name("variant_id")
        .ok_or_else(|| anyhow!("Record batch does not have a variant_id column.",))?
        .as_any()
        .downcast_ref::<LargeStringArray>()
        .ok_or_else(|| anyhow!("Couldn't cast variant_id column to StringArray.",))?;
    let betas = feature_column
        .column_by_name("beta")
        .ok_or_else(|| anyhow!("Feature column does not have a beta column."))?
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| anyhow!("Couldn't cast feature column - beta column to Float32Array."))?;
    let genotype_variances = feature_column
        .column_by_name("genotype_variance")
        .ok_or_else(|| anyhow!("Feature column does not have a genotype_variance column.",))?
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| {
            anyhow!("Couldn't cast feature column - genotype_variance column to Float32Array.",)
        })?;
    let degrees_of_freedom = feature_column
        .column_by_name("degrees_of_freedom")
        .ok_or_else(|| anyhow!("Feature column does not have a degrees_of_freedom column.",))?
        .as_any()
        .downcast_ref::<Int32Array>()
        .ok_or_else(|| {
            anyhow!("Couldn't cast feature column - degrees_of_freedom column to Int32Array.",)
        })?;

    Ok(FeatureStats {
        variant_id: variant_ids.clone(),
        beta: betas.clone(),
        genotype_variance: genotype_variances.clone(),
        degrees_of_freedom: degrees_of_freedom.clone(),
    })
}

pub fn compute_batch_stats_running(
    record_batch: &RecordBatch,
    projection: &Projection,
) -> Result<RunningStats> {
    let n_variants = record_batch.num_rows();
    let mut running_stats = RunningStats::new(n_variants);

    for (feature_id, projection_coefficient) in projection
        .feature_id
        .iter()
        .zip(projection.feature_coefficient.iter())
        .filter(|(_, &c)| c != 0.0)
    {
        let feature_stats = get_columns(record_batch, feature_id)?;

        for (i, (variant_id, beta, genotype_variance, degrees_of_freedom)) in izip!(
            feature_stats.variant_id.iter(),
            feature_stats.beta.iter(),
            feature_stats.genotype_variance.iter(),
            feature_stats.degrees_of_freedom.iter(),
        )
        .enumerate()
        {
            running_stats.variant_id[i] = variant_id.unwrap().to_string();
            running_stats.beta[i] += projection_coefficient * beta.unwrap();
            running_stats.genotype_variance[i] +=
                genotype_variance.unwrap() / projection.n_features as f32;
            running_stats.degrees_of_freedom[i] =
                running_stats.degrees_of_freedom[i].min(degrees_of_freedom.unwrap());
        }
    }

    Ok(running_stats)
}

/// Compute the minor allele frequency from the variance of the additively
/// coded genotype.
pub fn compute_maf_from_variance(genotype_variance: f32) -> f32 {
    (1.0 - (1.0 - 2.0 * genotype_variance).sqrt()) / 2.0
}

pub fn compute_batch_results(
    running_stats: RunningStats,
    projection_variance: f32,
    n_covariates: usize,
) -> Result<ResultStats> {
    let std_error: Vec<f32> = izip!(
        running_stats.beta.iter(),
        running_stats.genotype_variance.iter(),
        running_stats.degrees_of_freedom.iter()
    )
    .map(|(beta, genotype_variance, degrees_of_freedom)| {
        indirect_std_error(
            projection_variance,
            *genotype_variance,
            *beta,
            *degrees_of_freedom,
        )
    })
    .collect();

    let t_stat: Vec<f32> = running_stats
        .beta
        .iter()
        .zip(std_error.iter())
        .map(|(beta, std_error)| beta / std_error)
        .collect();

    let neg_log_p_value: Vec<f32> = t_stat
        .iter()
        .zip(running_stats.degrees_of_freedom.iter())
        .map(|(&t_stat, &dof)| igwas::stats::sumstats::compute_neg_log_pvalue(t_stat, dof))
        .collect();

    let sample_size: Vec<i32> = running_stats
        .degrees_of_freedom
        .iter()
        .map(|dof| dof + n_covariates as i32 + 2)
        .collect();

    let allele_frequency = running_stats
        .genotype_variance
        .iter()
        .map(|&gv| compute_maf_from_variance(gv))
        .collect();

    Ok(ResultStats {
        variant_id: running_stats.variant_id,
        beta: running_stats.beta,
        std_error,
        t_stat,
        neg_log_p_value,
        sample_size,
        allele_frequency,
    })
}

pub fn results_to_record_batch(result_stats: ResultStats) -> Result<RecordBatch> {
    let schema = Schema::new(vec![
        Field::new("variant_id", DataType::LargeUtf8, false),
        Field::new("beta", DataType::Float32, false),
        Field::new("std_error", DataType::Float32, false),
        Field::new("t_stat", DataType::Float32, false),
        Field::new("neg_log_p_value", DataType::Float32, false),
        Field::new("sample_size", DataType::Int32, false),
        Field::new("allele_frequency", DataType::Float32, false),
    ]);

    let record_batch = RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(LargeStringArray::from(result_stats.variant_id)),
            Arc::new(Float32Array::from(result_stats.beta)),
            Arc::new(Float32Array::from(result_stats.std_error)),
            Arc::new(Float32Array::from(result_stats.t_stat)),
            Arc::new(Float32Array::from(result_stats.neg_log_p_value)),
            Arc::new(Int32Array::from(result_stats.sample_size)),
            Arc::new(Float32Array::from(result_stats.allele_frequency)),
        ],
    )?;
    Ok(record_batch)
}

pub fn get_reader(path: &str, batch_size: usize) -> Result<ParquetRecordBatchReader> {
    let input_file = File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(input_file)?.with_batch_size(batch_size);
    Ok(builder.build()?)
}

pub fn get_writer(path: &str) -> Result<arrow::csv::Writer<AutoFinishEncoder<BufWriter<File>>>> {
    let file = File::create(path)?;
    let buffer_writer = BufWriter::new(file);
    let compressed_writer = zstd::Encoder::new(buffer_writer, 3)?.auto_finish();
    Ok(arrow::csv::WriterBuilder::new()
        .with_delimiter(b'\t')
        .build(compressed_writer))
}

/// Run Indirect GWAS
///
/// Indirect GWAS is a method for computing summary statistics for a GWAS on
/// a linear combination of phenotypes. For more information, see
/// https://github.com/tatonetti-lab/indirect-gwas.
///
/// # Arguments
///
/// * `projection` - Labeled projection vector
/// * `projection_variance` - The variance of the projected phenotype
/// * `n_covariates` - The number of covariates used in the GWAS
/// * `input_path` - Path to the input file (parquet)
/// * `output_path` - Path to the output file (tsv.zst)
#[pyfunction]
#[pyo3(signature = (projection, projection_variance, n_covariates, input_path, output_path, batch_size = 100000))]
pub fn run_igwas(
    projection: PyRef<'_, Projection>,
    projection_variance: f32,
    n_covariates: usize,
    input_path: &str,
    output_path: &str,
    batch_size: usize,
) -> PyResult<()> {
    let reader =
        get_reader(input_path, batch_size).map_err(|e| PyValueError::new_err(format!("{e}")))?;
    let mut writer = get_writer(output_path).map_err(|e| PyValueError::new_err(format!("{e}")))?;

    for record_batch in reader {
        match record_batch {
            Ok(record_batch) => {
                let running_stats = compute_batch_stats_running(&record_batch, &projection)
                    .map_err(|e| PyValueError::new_err(format!("{e}")))?;
                let result_stats =
                    compute_batch_results(running_stats, projection_variance, n_covariates)
                        .map_err(|e| PyValueError::new_err(format!("{e}")))?;
                let result_batch = results_to_record_batch(result_stats)
                    .map_err(|e| PyValueError::new_err(format!("{e}")))?;
                writer
                    .write(&result_batch)
                    .map_err(|e| PyValueError::new_err(format!("{e}")))?;
            }
            Err(e) => return Err(PyValueError::new_err(format!("{e}"))),
        }
    }
    Ok(())
}
