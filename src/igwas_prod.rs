use log::debug;
use std::{fs::File, io::BufWriter};

use anyhow::{anyhow, Context, Result};
use itertools::izip;
use polars::prelude::*;
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
    projection: &Projection,
) -> Result<RunningStats> {
    let n_variants = df.height();
    let mut running_stats = RunningStats::new(n_variants);

    for (feature_idx, (feature_id, projection_coefficient)) in projection
        .feature_id
        .iter()
        .zip(projection.feature_coefficient.iter())
        .filter(|(_, &c)| c != 0.0)
        .enumerate()
    {
        let feature_stats = get_columns(df, feature_id, feature_idx == 0)?;
        if let Some(variant_id) = feature_stats.variant_id {
            running_stats.variant_id = variant_id
        }
        for (i, (beta, genotype_variance, degrees_of_freedom)) in izip!(
            feature_stats.beta.iter(),
            feature_stats.genotype_variance.iter(),
            feature_stats.degrees_of_freedom.iter(),
        )
        .enumerate()
        {
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
    projection: &Projection,
    projection_variance: f32,
    n_covariates: usize,
    output_path: String,
    n_threads: usize,
) -> Result<()> {
    debug!("Computing batch stats");
    let running_stats = compute_batch_stats_running(gwas_df, projection)?;
    debug!("Computing batch results");
    let result_stats = compute_batch_results(running_stats, projection_variance, n_covariates)?;
    debug!("Converting results to dataframe");
    let mut results_df = results_to_dataframe(result_stats)?;
    debug!("Writing results");
    write_dataframe(&mut results_df, &output_path, n_threads)?;
    Ok(())
}
