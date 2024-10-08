use anyhow::{anyhow, Context, Result};
use itertools::izip;
use log::debug;
use ndarray::prelude::*;
use polars::prelude::*;
use rayon::prelude::*;
use std::{fs::File, io::BufWriter};
use zstd::stream::AutoFinishEncoder;

#[derive(Debug)]
pub struct Projection {
    pub feature_id: Vec<String>,
    pub feature_coefficient: Vec<f32>,
    pub n_features: usize,
}

impl Projection {
    pub fn new(feature_id: Vec<String>, feature_coefficient: Vec<f32>) -> Result<Self> {
        if feature_id.len() != feature_coefficient.len() {
            return Err(anyhow!(
                "feature_id and feature_coefficient must have the same length.",
            ));
        }
        let n_features = feature_coefficient
            .iter()
            .filter(|&&c| c != 0.0_f32)
            .count();
        let result = Projection {
            feature_id,
            feature_coefficient,
            n_features,
        };
        Ok(result)
    }

    pub fn remove_zeros(&mut self) {
        let mut n_features = 0;
        let mut new_feature_id = Vec::new();
        let mut new_feature_coefficient = Vec::new();
        for (feature_id, feature_coefficient) in
            self.feature_id.iter().zip(self.feature_coefficient.iter())
        {
            if feature_coefficient != &0.0 {
                new_feature_id.push(feature_id.clone());
                new_feature_coefficient.push(*feature_coefficient);
                n_features += 1;
            }
        }
        self.n_features = n_features;
        self.feature_id = new_feature_id;
        self.feature_coefficient = new_feature_coefficient;
    }

    pub fn standardize(&mut self, full_feature_ids: &[String]) {
        let mut new_feature_coefficient = Vec::new();
        for feature_id in full_feature_ids {
            match self.feature_id.contains(feature_id) {
                true => {
                    let original_index = self
                        .feature_id
                        .iter()
                        .position(|x| x == feature_id)
                        .unwrap();
                    new_feature_coefficient.push(self.feature_coefficient[original_index]);
                }
                false => {
                    new_feature_coefficient.push(0.0);
                }
            }
        }
        self.feature_coefficient = new_feature_coefficient;
        self.feature_id = full_feature_ids.to_vec();
        self.n_features = full_feature_ids.len();
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &f32)> {
        self.feature_id.iter().zip(self.feature_coefficient.iter())
    }

    pub fn par_iter(&self) -> impl ParallelIterator<Item = (&String, &f32)> {
        self.feature_id
            .par_iter()
            .zip(self.feature_coefficient.par_iter())
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
    pub beta: Vec<f32>,
    pub genotype_variance: Vec<f32>,
    pub degrees_of_freedom: Vec<i32>,
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

pub struct FeatureStats {
    pub variant_id: Option<Vec<String>>,
    pub beta: Float32Chunked,
    pub genotype_variance: Float32Chunked,
    pub degrees_of_freedom: Int32Chunked,
}

pub fn compute_batch_stats(df: &DataFrame, projection: &mut Projection) -> Result<RunningStats> {
    let (gwas_beta, feature_ids) = {
        let gwas_beta_df = df.drop_many(["variant_id", "genotype_variance", "degrees_of_freedom"]);
        let feature_ids = gwas_beta_df
            .get_column_names()
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<String>>();
        let gwas_beta = gwas_beta_df.to_ndarray::<Float32Type>(IndexOrder::Fortran)?;
        (gwas_beta, feature_ids)
    };
    projection.standardize(&feature_ids);
    let projection_coefs = Array1::from_vec(projection.feature_coefficient.clone());
    let gwas_beta = gwas_beta.dot(&projection_coefs).to_vec();
    let variant_id = df
        .column("variant_id")
        .context("No variant_id column")?
        .str()
        .context("variant_id column is not str")?
        .iter()
        .map(|x| x.expect("Failed to get variant_id").to_string())
        .collect::<Vec<String>>();
    let genotype_variance = df
        .column("genotype_variance")
        .context("No genotype_variance column")?
        .f32()
        .context("genotype_variance column is not f32")?
        .iter()
        .collect::<Option<Vec<f32>>>()
        .expect("Failed to collect genotype_variance");
    let degrees_of_freedom = df
        .column("degrees_of_freedom")
        .context("No degrees_of_freedom column")?
        .i32()
        .context("Degrees_of_freedom column is not i32")?
        .iter()
        .collect::<Option<Vec<i32>>>()
        .expect("Failed to collect degrees_of_freedom");
    Ok(RunningStats {
        variant_id,
        beta: gwas_beta,
        genotype_variance,
        degrees_of_freedom,
    })
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
        &running_stats.beta,
        &running_stats.genotype_variance,
        &running_stats.degrees_of_freedom
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

pub fn results_to_dataframe(result_stats: ResultStats) -> Result<DataFrame> {
    let df = DataFrame::new(vec![
        Series::new("variant_id".into(), result_stats.variant_id),
        Series::new("beta".into(), result_stats.beta),
        Series::new("std_error".into(), result_stats.std_error),
        Series::new("t_stat".into(), result_stats.t_stat),
        Series::new("neg_log_p_value".into(), result_stats.neg_log_p_value),
        Series::new("sample_size".into(), result_stats.sample_size),
        Series::new("allele_frequency".into(), result_stats.allele_frequency),
    ])?;
    Ok(df)
}

pub fn read_dataframe(path: &str) -> Result<DataFrame> {
    let input_file = File::open(path)?;
    let df = ParquetReader::new(input_file).finish()?;
    Ok(df)
}

pub fn get_writer(path: &str) -> Result<arrow::csv::Writer<AutoFinishEncoder<BufWriter<File>>>> {
    let file = File::create(path)?;
    let buffer_writer = BufWriter::new(file);
    let compressed_writer = zstd::Encoder::new(buffer_writer, 3)?.auto_finish();
    Ok(arrow::csv::WriterBuilder::new()
        .with_delimiter(b'\t')
        .build(compressed_writer))
}

pub fn write_dataframe(df: &mut DataFrame, path: &str, n_threads: usize) -> Result<()> {
    let file = File::create(path)?;
    let compressed_writer = zstd::Encoder::new(file, 3)?.auto_finish();
    CsvWriter::new(compressed_writer)
        .with_separator(b'\t')
        .n_threads(n_threads)
        .finish(df)?;
    Ok(())
}

pub fn run_igwas_df_impl(
    gwas_df: &DataFrame,
    projection: &mut Projection,
    projection_variance: f32,
    n_covariates: usize,
    output_path: String,
    n_threads: usize,
) -> Result<()> {
    debug!("Computing batch stats");
    let running_stats = compute_batch_stats(gwas_df, projection)?;
    debug!("Computing batch results");
    let result_stats = compute_batch_results(running_stats, projection_variance, n_covariates)?;
    debug!("Converting results to dataframe");
    let mut results_df = results_to_dataframe(result_stats)?;
    debug!("Writing results");
    write_dataframe(&mut results_df, &output_path, n_threads)?;
    Ok(())
}
