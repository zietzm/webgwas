use crate::atomics::AtomicF32;
use anyhow::{anyhow, Context, Result};
use core::sync::atomic::Ordering;
use itertools::izip;
use log::debug;
use polars::prelude::*;
use rayon::prelude::*;
use std::sync::atomic::AtomicI32;
use std::sync::Mutex;
use std::{fs::File, io::BufWriter};
use zstd::stream::AutoFinishEncoder;

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

pub struct RunningStatsAtomic {
    pub variant_id: Mutex<Vec<String>>,
    pub n_variants: usize,
    pub beta: Vec<AtomicF32>,
    pub genotype_variance: Vec<AtomicF32>,
    pub degrees_of_freedom: Vec<AtomicI32>,
}

pub struct RunningStats {
    pub variant_id: Vec<String>,
    pub beta: Vec<f32>,
    pub genotype_variance: Vec<f32>,
    pub degrees_of_freedom: Vec<i32>,
}

impl RunningStatsAtomic {
    pub fn new(n_variants: usize) -> Self {
        RunningStatsAtomic {
            variant_id: Mutex::new(Vec::new()),
            n_variants,
            beta: (0..n_variants).map(|_| AtomicF32::new(0.0)).collect(),
            genotype_variance: (0..n_variants).map(|_| AtomicF32::new(0.0)).collect(),
            degrees_of_freedom: (0..n_variants).map(|_| AtomicI32::new(i32::MAX)).collect(),
        }
    }

    pub fn de_atomize(self) -> RunningStats {
        RunningStats {
            variant_id: self.variant_id.lock().unwrap().to_vec(),
            beta: self
                .beta
                .into_iter()
                .map(|b| b.load(Ordering::Relaxed))
                .collect(),
            genotype_variance: self
                .genotype_variance
                .into_iter()
                .map(|gv| gv.load(Ordering::Relaxed))
                .collect(),
            degrees_of_freedom: self
                .degrees_of_freedom
                .into_iter()
                .map(|dof| dof.load(Ordering::Relaxed))
                .collect(),
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

pub struct FeatureStats {
    pub variant_id: Option<Vec<String>>,
    pub beta: Float32Chunked,
    pub genotype_variance: Float32Chunked,
    pub degrees_of_freedom: Int32Chunked,
}

fn get_columns(df: &DataFrame, feature_id: &str, needs_variant_id: bool) -> Result<FeatureStats> {
    let feature_column_df = df
        .column(feature_id)
        .with_context(|| anyhow!("Feature column not found."))?
        .struct_()
        .with_context(|| anyhow!("Feature column {} is not a struct", feature_id))?
        .clone()
        .unnest();
    let betas = feature_column_df
        .column("beta")
        .with_context(|| anyhow!("Feature column does not have a beta column."))?
        .f32()
        .with_context(|| anyhow!("Couldn't cast feature column - beta column to Float32Array."))?;
    let genotype_variances = feature_column_df
        .column("genotype_variance")
        .with_context(|| anyhow!("Feature column does not have a genotype_variance column."))?
        .f32()
        .with_context(|| {
            anyhow!("Couldn't cast feature column - genotype_variance column to Float32Array.")
        })?;
    let degrees_of_freedom = feature_column_df
        .column("degrees_of_freedom")
        .with_context(|| anyhow!("Feature column does not have a degrees_of_freedom column."))?
        .i32()
        .with_context(|| {
            anyhow!("Couldn't cast feature column - degrees_of_freedom column to Int32Array.")
        })?;
    let variant_ids = if needs_variant_id {
        let variant_ids = df
            .column("variant_id")
            .with_context(|| anyhow!("Feature column does not have a variant_id column."))?
            .str()
            .with_context(|| anyhow!("Couldn't cast variant_id column to StringArray."))?
            .iter()
            .map(|s| s.unwrap().to_string())
            .collect::<Vec<String>>();
        Some(variant_ids)
    } else {
        None
    };
    Ok(FeatureStats {
        variant_id: variant_ids,
        beta: betas.clone(),
        genotype_variance: genotype_variances.clone(),
        degrees_of_freedom: degrees_of_freedom.clone(),
    })
}

pub fn compute_batch_stats_running(
    df: &DataFrame,
    projection: &mut Projection,
) -> Result<RunningStatsAtomic> {
    let n_variants = df.height();
    let running_stats = Arc::new(RunningStatsAtomic::new(n_variants));
    projection.remove_zeros();
    let first_feature_id = projection.feature_id[0].clone();
    let is_first_feature = |feature_id: &String| *feature_id == first_feature_id;

    projection
        .par_iter()
        .for_each(|(feature_id, projection_coefficient)| {
            let feature_stats = get_columns(df, feature_id, is_first_feature(feature_id)).unwrap();
            if is_first_feature(feature_id) {
                let mut state = running_stats
                    .variant_id
                    .lock()
                    .expect("Failed to lock variant_id");
                *state = feature_stats.variant_id.expect("Failed to get variant_id");
            }
            for i in 0..feature_stats.beta.len() {
                running_stats.beta[i]
                    .add(projection_coefficient * feature_stats.beta.get(i).unwrap());
                running_stats.genotype_variance[i].add(
                    feature_stats.genotype_variance.get(i).unwrap() / projection.n_features as f32,
                );
                running_stats.degrees_of_freedom[i].fetch_min(
                    feature_stats.degrees_of_freedom.get(i).unwrap(),
                    Ordering::Relaxed,
                );
            }
        });
    Arc::try_unwrap(running_stats)
        .ok()
        .context("Failed to unwrap running stats")
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
        Series::new("variant_id", result_stats.variant_id),
        Series::new("beta", result_stats.beta),
        Series::new("std_error", result_stats.std_error),
        Series::new("t_stat", result_stats.t_stat),
        Series::new("neg_log_p_value", result_stats.neg_log_p_value),
        Series::new("sample_size", result_stats.sample_size),
        Series::new("allele_frequency", result_stats.allele_frequency),
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
    let atomic_running_stats = compute_batch_stats_running(gwas_df, projection)?;
    debug!("De-atomizing running stats");
    let running_stats = atomic_running_stats.de_atomize();
    debug!("Computing batch results");
    let result_stats = compute_batch_results(running_stats, projection_variance, n_covariates)?;
    debug!("Converting results to dataframe");
    let mut results_df = results_to_dataframe(result_stats)?;
    debug!("Writing results");
    write_dataframe(&mut results_df, &output_path, n_threads)?;
    Ok(())
}
