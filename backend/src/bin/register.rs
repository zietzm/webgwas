use anyhow::{bail, Context, Result};
use clap::Parser;
use faer::Mat;
use faer_ext::polars::polars_to_faer_f32;
use indicatif::{ProgressBar, ProgressState, ProgressStyle};
use log::{error, info, warn};
use mdav::mdav::mdav;
use polars::lazy::dsl::{mean_horizontal, min_horizontal};
use polars::prelude::*;
use rusqlite::OptionalExtension;
use std::ops::Sub;
use std::{
    collections::{HashMap, HashSet},
    fmt::Write,
    fs::File,
    io::BufWriter,
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};

use webgwas_backend::{
    models::{Cohort, CohortMetadata},
    regression::{
        add_intercept, compute_covariance, compute_left_inverse, residualize_covariates,
        transpose_vec_vec,
    },
};

fn main() {
    env_logger::init();
    let args = Cli::parse();
    let mut app_state = LocalAppState::new(args.cohort_name.clone(), args.overwrite)
        .context("Failed to initialize app state")
        .unwrap();
    let result = app_state.register_cohort(
        args.pheno_file,
        args.covar_file,
        args.gwas_files,
        args.feature_info_file,
        args.gwas_spec,
        args.pheno_spec,
    );
    match result {
        Ok(_) => {
            info!("Successfully registered cohort '{}'", args.cohort_name);
        }
        Err(err) => {
            error!("Failed to register cohort: {}", err);
            // app_state
            //     .cleanup()
            //     .context("Failed to cleanup app state")
            //     .unwrap();
        }
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Cohort name (mandatory)
    #[arg(long, required = true)]
    cohort_name: String,

    // Phenotype file arguments
    #[command(flatten)]
    pheno_file: PhenoFile,

    // Covariate file arguments (optional)
    #[command(flatten)]
    covar_file: CovarFile,

    /// GWAS files
    #[arg(long = "gwas-files", num_args = 1..)]
    gwas_files: Vec<PathBuf>,

    /// File giving information about each feature
    #[command(flatten)]
    feature_info_file: FeatureInfoFile,

    // Options for processing GWAS files
    #[command(flatten)]
    gwas_spec: GWASOptions,

    // Options for processing phenotype files
    #[command(flatten)]
    pheno_spec: PhenoOptions,

    // Overwrite existing cohort
    #[arg(long = "overwrite", default_value_t = false)]
    overwrite: bool,
}

#[derive(Parser, Debug)]
pub struct PhenoFile {
    /// Phenotype file path (mandatory)
    #[arg(long = "pheno-path", required = true)]
    pheno_path: PathBuf,

    /// Phenotype file separator
    #[arg(long = "pheno-separator", default_value_t = b'\t')]
    pheno_separator: u8,

    /// Phenotype file person ID column (mandatory)
    #[arg(long = "pheno-person-id-column", default_value = "eid")]
    pheno_person_id_column: String,

    /// Phenotype file ignore columns
    #[arg(long = "pheno-ignore-columns")]
    pheno_ignore_columns: Option<Vec<String>>,
}

#[derive(Parser, Debug)]
pub struct CovarFile {
    /// Covariate file path
    #[arg(long = "covar-path", required = true)]
    covar_path: PathBuf,

    /// Covariate file separator (requires covar-path)
    #[arg(
        long = "covar-separator",
        requires = "covar_path",
        default_value_t = b'\t'
    )]
    covar_separator: u8,

    /// Covariate file person ID column (requires covar-path)
    #[arg(
        long = "covar-person-id-column",
        requires = "covar_path",
        default_value = "eid"
    )]
    covar_person_id_column: String,

    /// Covariate file ignore columns (requires covar-path)
    #[arg(long = "covar-ignore-columns", requires = "covar_path")]
    covar_ignore_columns: Option<Vec<String>>,
}

#[derive(Parser, Debug)]
pub struct FeatureInfoFile {
    /// Feature info file path (mandatory)
    #[arg(long = "feature-info-path", required = true)]
    feature_info_path: PathBuf,

    /// Feature info file separator
    #[arg(long = "feature-info-separator", default_value_t = b'\t')]
    feature_info_separator: u8,
}

#[derive(Parser, Debug)]
pub struct GWASOptions {
    /// GWAS file separator
    #[arg(long = "gwas-separator", default_value_t = b'\t')]
    pub gwas_separator: u8,

    /// GWAS variant ID column
    #[arg(long = "variant-id-column", default_value = "ID")]
    pub variant_id_column: String,

    /// GWAS beta column
    #[arg(long = "beta-column", default_value = "BETA")]
    pub beta_column: String,

    /// GWAS standard error column
    #[arg(long = "std-err-column", default_value = "SE")]
    pub std_err_column: String,

    /// GWAS sample size column
    #[arg(long = "sample-size-column", default_value = "OBS_CT")]
    pub sample_size_column: String,

    /// Keep n variants
    #[arg(long = "keep-n-variants")]
    pub keep_n_variants: Option<usize>,
}

#[derive(Parser, Debug)]
pub struct PhenoOptions {
    /// k-anonymity
    #[arg(long = "k-anonymity", default_value = "10")]
    k_anonymity: usize,

    /// Keep n samples
    #[arg(long = "keep-n-samples")]
    keep_n_samples: Option<usize>,
}

pub struct LocalAppState {
    pub root_directory: PathBuf,
    pub cohort_directory: PathBuf,
    pub db: rusqlite::Connection,
    pub feature_sets: FeatureSets,
    pub metadata: CohortMetadata,
    pub feature_variance: HashMap<String, f32>,
    pub overwrite: bool,
}

#[derive(Debug, Default)]
pub struct FeatureSets {
    pub phenotype_file: HashSet<String>,
    pub gwas_files: HashSet<String>,
    pub info_file: HashSet<String>,
    pub gwas_feature_to_file: HashMap<String, PathBuf>,
    pub final_features: HashSet<String>,
}

impl LocalAppState {
    pub fn new(cohort_name: String, overwrite: bool) -> Result<Self> {
        let home = std::env::var("HOME").expect("Failed to read $HOME");
        let root_directory = PathBuf::from(home).join("webgwas");
        if !root_directory.exists() {
            std::fs::create_dir_all(&root_directory)?;
        }
        let sqlite_db_path = root_directory.join("webgwas.db");
        let db = initialize_database(&sqlite_db_path)?;
        let feature_sets = FeatureSets::default();
        let metadata = CohortMetadata {
            cohort_id: None,
            cohort_name: cohort_name.clone(),
            num_covar: None,
        };
        let normalized_cohort_name = normalize_cohort_name(&cohort_name);
        let cohort_directory = root_directory.join("cohorts").join(&normalized_cohort_name);
        Ok(Self {
            root_directory,
            cohort_directory,
            db,
            feature_sets,
            metadata,
            feature_variance: HashMap::new(),
            overwrite,
        })
    }

    pub fn register_cohort(
        &mut self,
        pheno_file_spec: PhenoFile,
        covar_file_spec: CovarFile,
        gwas_files: Vec<PathBuf>,
        feature_info_file: FeatureInfoFile,
        gwas_options: GWASOptions,
        pheno_options: PhenoOptions,
    ) -> Result<()> {
        info!("Registering cohort {}", self.metadata.cohort_name);
        match self.check_cohort_does_not_exist() {
            Ok(_) => {
                std::fs::create_dir_all(&self.cohort_directory)?;
            }
            Err(_) => {
                if self.overwrite {
                    warn!(
                        "Cohort {} already exists, overwriting",
                        self.metadata.cohort_name
                    );
                    std::fs::remove_dir_all(&self.cohort_directory)?;
                    std::fs::create_dir_all(&self.cohort_directory)?;
                } else {
                    bail!("Cohort {} already exists", self.metadata.cohort_name);
                }
            }
        }
        self.find_usable_features(
            &pheno_file_spec,
            &covar_file_spec,
            &gwas_files,
            &feature_info_file,
        )?;
        self.process_phenotypes_covariates(&pheno_file_spec, &covar_file_spec, pheno_options)?;
        self.process_gwas_files(gwas_options)?;

        // TODO: Process the feature info file
        self.process_feature_info_file(&feature_info_file)?;

        bail!("Unimplement register_cohort")
    }

    pub fn check_cohort_does_not_exist(&mut self) -> Result<()> {
        check_cohort_not_in_db(&self.metadata.cohort_name, &self.db)?;
        if self.cohort_directory.exists() {
            bail!("Cohort {} already exists", self.metadata.cohort_name);
        }
        Ok(())
    }

    pub fn find_usable_features(
        &mut self,
        pheno_file_spec: &PhenoFile,
        covar_file_spec: &CovarFile,
        gwas_files: &[PathBuf],
        feature_info_file: &FeatureInfoFile,
    ) -> Result<()> {
        // Find features in the phenotype file and count the number of covariates
        info!("Registering phenotypes and covariates");
        self.register_phenotypes_covariates(pheno_file_spec, covar_file_spec)?;

        // Find features in the GWAS files
        info!("Registering GWAS files");
        self.register_gwas_files(gwas_files)?;

        // Find features in the feature info file
        info!("Registering feature info file");
        self.register_feature_info(feature_info_file)?;

        // Find the shared features
        self.feature_sets.final_features = self
            .feature_sets
            .phenotype_file
            .intersection(&self.feature_sets.gwas_files)
            .cloned()
            .collect();
        self.feature_sets.final_features = self
            .feature_sets
            .final_features
            .intersection(&self.feature_sets.info_file)
            .cloned()
            .collect();
        let num_final_features = self.feature_sets.final_features.len();
        info!("Found {} shared features", num_final_features);
        Ok(())
    }

    pub fn register_phenotypes_covariates(
        &mut self,
        pheno_file_spec: &PhenoFile,
        covar_file_spec: &CovarFile,
    ) -> Result<()> {
        let num_covar = get_num_covar(covar_file_spec)?;
        self.metadata.num_covar = Some(num_covar);

        // Get all the phenotype names
        let mut ignore_columns = pheno_file_spec
            .pheno_ignore_columns
            .clone()
            .unwrap_or_default();
        ignore_columns.push(pheno_file_spec.pheno_person_id_column.clone());
        let phenotype_names = read_delim_header(
            &pheno_file_spec.pheno_path,
            pheno_file_spec.pheno_separator,
            Some(ignore_columns),
        )?;
        self.feature_sets.phenotype_file = phenotype_names.into_iter().collect();
        Ok(())
    }

    pub fn register_gwas_files(&mut self, gwas_files: &[PathBuf]) -> Result<()> {
        for gwas_file in gwas_files {
            let feature_name = get_gwas_feature_name(gwas_file);
            match self.feature_sets.gwas_files.insert(feature_name.clone()) {
                true => {
                    self.feature_sets
                        .gwas_feature_to_file
                        .insert(feature_name, gwas_file.clone());
                }
                false => bail!("GWAS file {} already exists", gwas_file.display()),
            }
        }
        Ok(())
    }

    pub fn register_feature_info(&mut self, feature_info_file: &FeatureInfoFile) -> Result<()> {
        let parse_opts =
            CsvParseOptions::default().with_separator(feature_info_file.feature_info_separator);
        let codes = CsvReadOptions::default()
            .with_columns(Some(Arc::new(["code".into()])))
            .with_parse_options(parse_opts)
            .try_into_reader_with_file_path(Some(feature_info_file.feature_info_path.clone()))
            .context("Failed to set feature info file as path")?
            .finish()?["code"]
            .str()?
            .into_iter()
            .map(|x_opt| x_opt.unwrap().to_string())
            .collect::<Vec<String>>();
        for code in codes.iter() {
            match self.feature_sets.info_file.insert(code.clone()) {
                true => {}
                false => bail!("Feature info file contains duplicate code {}", code),
            }
        }
        Ok(())
    }

    pub fn process_phenotypes_covariates(
        &mut self,
        pheno_file_spec: &PhenoFile,
        covar_file_spec: &CovarFile,
        pheno_options: PhenoOptions,
    ) -> Result<()> {
        info!("Reading phenotypes and covariates");
        let pheno_cols = self
            .feature_sets
            .final_features
            .iter()
            .cloned()
            .collect::<Vec<String>>();
        let start = Instant::now();
        let data = read_phenotypes_covariates(&pheno_cols, pheno_file_spec, covar_file_spec, true)?;
        let duration = start.elapsed();
        info!("Reading phenotypes and covariates took {:?}", duration);

        info!("Computing partial covariance matrix");
        let start = Instant::now();
        let y_resid = residualize_covariates(&data.x_covariates, &data.y_phenotypes)?;
        let covariance_mat = compute_covariance(&y_resid, 1); // TODO: What is correct DDOF?
        let duration = start.elapsed();
        info!("Computing partial covariance matrix took {:?}", duration);
        self.feature_variance = self
            .feature_sets
            .final_features
            .iter()
            .enumerate()
            .map(|(i, x)| (x.clone(), covariance_mat[(i, i)]))
            .collect();
        let start = Instant::now();
        let mut covariance_df = mat_to_polars(covariance_mat, &data.phenotype_names)?;
        let duration = start.elapsed();
        info!("Converting covariance matrix took {:?}", duration);
        let start = Instant::now();
        let covar_path = self.cohort_directory.join("covariance.parquet");
        write_parquet(&mut covariance_df, &covar_path)?;
        let duration = start.elapsed();
        info!("Writing covariance matrix took {:?}", duration);

        info!("Anonymizing phenotypes");
        let start = Instant::now();
        let n_samples = match pheno_options.keep_n_samples {
            Some(n) => n,
            None => data.y_phenotypes.nrows(),
        };
        let mut raw_phenotypes_with_intercept = data.y_phenotypes.clone();
        add_intercept(&mut raw_phenotypes_with_intercept);
        let raw_phenotypes = raw_phenotypes_with_intercept
            .row_iter()
            .take(n_samples)
            .map(|x| x.iter().copied().collect::<Vec<f32>>())
            .collect::<Vec<Vec<f32>>>();
        let anonymized_phenotypes = mdav(raw_phenotypes, pheno_options.k_anonymity);
        let duration = start.elapsed();
        info!("Anonymizing phenotypes took {:?}", duration);
        let mut phenotype_colnames = data.phenotype_names.clone();
        phenotype_colnames.push("intercept".to_string());
        let start = Instant::now();
        let mut anonymized_phenotypes_df =
            vec_vec_to_polars(anonymized_phenotypes.clone(), &phenotype_colnames)?;
        let pheno_path = self.cohort_directory.join("phenotypes.parquet");
        write_parquet(&mut anonymized_phenotypes_df, &pheno_path)?;
        let duration = start.elapsed();
        info!("Writing anonymized phenotypes took {:?}", duration);

        info!("Computing left inverse");
        let start = Instant::now();
        let anonymized_phenotypes = vec_vec_to_mat(anonymized_phenotypes);
        let left_inverse = compute_left_inverse(&anonymized_phenotypes)?
            .transpose()
            .to_owned();
        let duration = start.elapsed();
        info!("Computing left inverse took {:?}", duration);
        let start = Instant::now();
        let mut left_inverse_df = mat_to_polars(left_inverse, &phenotype_colnames)?;
        let left_inverse_path = self.cohort_directory.join("phenotype_left_inverse.parquet");
        write_parquet(&mut left_inverse_df, &left_inverse_path)?;
        let duration = start.elapsed();
        info!("Writing left inverse took {:?}", duration);
        Ok(())
    }

    pub fn process_gwas_files(&mut self, gwas_options: GWASOptions) -> Result<()> {
        let variant_id = gwas_options.variant_id_column.clone();
        let beta = gwas_options.beta_column.clone();
        let std_err = gwas_options.std_err_column.clone();
        let sample_size = gwas_options.sample_size_column.clone();
        let column_specs = Arc::new([
            variant_id.clone().into(),
            beta.clone().into(),
            std_err.clone().into(),
            sample_size.clone().into(),
        ]);
        let schema_val: Vec<(PlSmallStr, DataType)> = vec![
            (variant_id.clone().into(), DataType::String),
            (beta.clone().into(), DataType::Float32),
            (std_err.clone().into(), DataType::Float32),
            (sample_size.clone().into(), DataType::Float32),
        ];
        let schema = Arc::new(Schema::from_iter(schema_val));
        let parse_opts = CsvParseOptions::default().with_separator(gwas_options.gwas_separator);
        let mut result_df = None;

        info!("Reading GWAS files");
        let pb = ProgressBar::new(self.feature_sets.final_features.len() as u64);
        pb.set_style(
            ProgressStyle::with_template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos:>7}/{len:7} ({eta})")
            .unwrap()
            .with_key("eta", |state: &ProgressState, w: &mut dyn Write| write!(w, "{:.1}s", state.eta().as_secs_f64()).unwrap())
            .progress_chars("#>-")
        );
        for (feature_name, gwas_file) in self.feature_sets.gwas_feature_to_file.iter() {
            let this_gwas_df = CsvReadOptions::default()
                .with_parse_options(parse_opts.clone())
                .with_columns(Some(column_specs.clone()))
                .with_schema_overwrite(Some(schema.clone()))
                .try_into_reader_with_file_path(Some(gwas_file.clone()))?
                .finish()
                .context(format!("Failed to read GWAS file {}", gwas_file.display()))?
                .lazy()
                .select([
                    col(&variant_id).alias("variant_id"),
                    col(&beta)
                        .cast(DataType::Float32)
                        .alias(format!("{}_beta", feature_name).as_str()),
                    col(&std_err)
                        .cast(DataType::Float32)
                        .alias(format!("{}_se", feature_name).as_str()),
                    col(&sample_size)
                        .cast(DataType::Float32)
                        .alias(format!("{}_ss", feature_name).as_str()),
                ])
                .collect()
                .context(format!("Failed to read GWAS file {}", gwas_file.display()))?;
            match result_df {
                None => {
                    result_df = Some(this_gwas_df);
                }
                Some(df) => {
                    let joined_df = df.inner_join(&this_gwas_df, ["variant_id"], ["variant_id"])?;
                    result_df = Some(joined_df);
                }
            }
            pb.inc(1);
        }
        pb.finish_with_message("Done reading GWAS files");
        let result_df = result_df.expect("Failed to collect result df");
        info!("Processing loaded GWAS dataframe");
        let start = Instant::now();
        let mut lazy_result_df = result_df.lazy().select([
            col("variant_id"),
            min_horizontal([col("^*_ss$")])?
                .sub(lit(self.metadata.num_covar.unwrap_or(0) as f32 + 2.0))
                .alias("degrees_of_freedom"),
            col("^*_beta$"),
            col("^*_se$"),
        ]);
        for feature_name in self.feature_sets.final_features.iter() {
            let feature_variance = self
                .feature_variance
                .get(feature_name)
                .expect("Failed to get feature variance");
            lazy_result_df = lazy_result_df.with_column(
                estimate_genotype_variance(
                    lit(*feature_variance),
                    col("degrees_of_freedom"),
                    col(format!("{}_beta", feature_name).as_str()),
                    col(format!("{}_se", feature_name).as_str()),
                )
                .alias(format!("{}_gvar", feature_name).as_str()),
            );
        }
        let mut final_result_df = lazy_result_df
            .select([
                col("variant_id"),
                col("degrees_of_freedom"),
                mean_horizontal([col("^*_gvar$")])?.alias("gvar"),
                col("^*_beta$")
                    .name()
                    .map(|x| Ok(x.replace("_beta", "").into())),
            ])
            .collect()?;
        let duration = start.elapsed();
        info!("Processing loaded GWAS dataframe took {:?}", duration);
        info!("Writing GWAS dataframe");
        let start = Instant::now();
        let gwas_path = self.cohort_directory.join("gwas.parquet");
        write_parquet(&mut final_result_df, &gwas_path)?;
        let duration = start.elapsed();
        info!("Writing GWAS dataframe took {:?}", duration);
        Ok(())
    }

    pub fn process_feature_info_file(&mut self, feature_info_file: &FeatureInfoFile) -> Result<()> {
        // TODO: Load the feature info file, write it to the database
        // TODO: Compute the sample size for each feature in advance, then write that to the DB as
        // well!
        bail!("Unimplement process_feature_info_file")
    }

    pub fn cleanup(&self) -> Result<()> {
        self.db.execute(
            "DELETE FROM cohort WHERE name = ?",
            [&self.metadata.cohort_name],
        )?;
        self.db.execute(
            "DELETE FROM phenotype WHERE cohort_id = ?",
            [&self.metadata.cohort_id],
        )?;
        std::fs::remove_dir_all(&self.cohort_directory)?;
        info!("Cleaned up all cohort data");
        Ok(())
    }
}

pub fn normalize_cohort_name(cohort_name: &str) -> String {
    cohort_name.to_lowercase().replace(" ", "_")
}

pub fn initialize_database(path: &Path) -> Result<rusqlite::Connection> {
    let db = rusqlite::Connection::open(path)?;
    db.execute(
        "CREATE TABLE IF NOT EXISTS cohort (
            id INTEGER NOT NULL,
            name VARCHAR NOT NULL,
            root_directory VARCHAR NOT NULL,
            num_covar INTEGER NOT NULL,
            PRIMARY KEY (id),
            UNIQUE (name),
            UNIQUE (root_directory)
        )",
        (),
    )?;
    db.execute(
        "CREATE TABLE IF NOT EXISTS phenotype (
                id INTEGER NOT NULL,
                code VARCHAR NOT NULL,
                name VARCHAR NOT NULL,
                type VARCHAR(4) NOT NULL,
                cohort_id INTEGER,
                PRIMARY KEY (id),
                CONSTRAINT unique_feature UNIQUE (cohort_id, code, name),
                FOREIGN KEY(cohort_id) REFERENCES cohort (id)
        )",
        (),
    )?;
    Ok(db)
}

/// Check if the cohort already exists in the database
pub fn check_cohort_not_in_db(cohort_name: &str, db: &rusqlite::Connection) -> Result<()> {
    let mut stmt = db.prepare(
        "SELECT id, name, root_directory, num_covar
            FROM cohort WHERE name = ?",
    )?;
    let existing_cohort = stmt
        .query_row([cohort_name], |row| {
            Ok(Cohort {
                id: row.get(0)?,
                name: row.get(1)?,
                root_directory: row.get(2)?,
                num_covar: row.get(3)?,
            })
        })
        .optional()?;
    if let Some(cohort) = existing_cohort {
        bail!("Cohort {} already exists", cohort.name);
    }
    Ok(())
}

/// Read the number of covariates
pub fn get_num_covar(covar_file_spec: &CovarFile) -> Result<i32> {
    let mut ignore_columns = covar_file_spec
        .covar_ignore_columns
        .clone()
        .unwrap_or_default();
    ignore_columns.push(covar_file_spec.covar_person_id_column.clone());
    let header = read_delim_header(
        &covar_file_spec.covar_path,
        covar_file_spec.covar_separator,
        Some(ignore_columns),
    )?;
    let num_covar = header.len() as i32;
    Ok(num_covar)
}

/// Read the header of a delimited file
pub fn read_delim_header(
    path: &Path,
    separator: u8,
    ignore_columns: Option<Vec<String>>,
) -> Result<Vec<String>> {
    let parse_opts = CsvParseOptions::default().with_separator(separator);
    let mut df = CsvReadOptions::default()
        .with_has_header(true)
        .with_n_rows(Some(0))
        .with_parse_options(parse_opts)
        .try_into_reader_with_file_path(Some(path.to_path_buf()))
        .context("Failed to set covariate file as path")?
        .finish()
        .context("Failed to read covariate file header")?;
    if let Some(ignore_columns) = ignore_columns {
        df = df.drop_many(ignore_columns);
    }
    let names = df
        .get_column_names()
        .iter()
        .map(|x| x.to_string())
        .collect();
    Ok(names)
}

/// Extract the phenotype name from a GWAS file path (e.g. /path/to/file.tsv.zst -> file)
pub fn get_gwas_feature_name(gwas_file: &Path) -> String {
    let file_name = gwas_file.file_name().unwrap().to_str().unwrap();
    let feature_name = file_name.split('.').next().unwrap();
    feature_name.to_string()
}

pub struct PhenotypesCovariates {
    pub y_phenotypes: Mat<f32>,
    pub x_covariates: Mat<f32>,
    pub phenotype_names: Vec<String>,
}

pub fn read_phenotypes_covariates(
    pheno_cols: &[String],
    pheno_file_spec: &PhenoFile,
    covar_file_spec: &CovarFile,
    add_intercept: bool,
) -> Result<PhenotypesCovariates> {
    let pheno_df = CsvReadOptions::default()
        .with_parse_options(
            CsvParseOptions::default().with_separator(pheno_file_spec.pheno_separator),
        )
        .try_into_reader_with_file_path(Some(pheno_file_spec.pheno_path.clone()))
        .context("Failed to set phenotype file as path")?
        .finish()?;
    let covar_df = CsvReadOptions::default()
        .with_parse_options(
            CsvParseOptions::default().with_separator(covar_file_spec.covar_separator),
        )
        .try_into_reader_with_file_path(Some(covar_file_spec.covar_path.clone()))
        .context("Failed to set covariate file as path")?
        .finish()?;
    info!("Merging phenotype and covariate files");
    let merged_df = pheno_df.join(
        &covar_df,
        [pheno_file_spec.pheno_person_id_column.as_str()],
        [covar_file_spec.covar_person_id_column.as_str()],
        JoinArgs::new(JoinType::Inner),
    )?;
    let pheno_cols = pheno_df
        .select(pheno_cols)?
        .get_column_names()
        .into_iter()
        .map(|x| x.to_string())
        .collect::<Vec<String>>();
    let y_df = merged_df.select(pheno_cols.clone())?.lazy();
    let y = polars_to_faer_f32(y_df)?;
    let covar_cols = covar_df
        .drop(covar_file_spec.covar_person_id_column.as_str())?
        .get_column_names()
        .into_iter()
        .map(|x| col(x.to_string()))
        .collect::<Vec<Expr>>();
    let mut x_lazy = merged_df.lazy().select(covar_cols);
    if add_intercept {
        x_lazy = x_lazy.with_column(lit(1.0).alias("intercept"));
    }
    let x = polars_to_faer_f32(x_lazy)?;
    Ok(PhenotypesCovariates {
        y_phenotypes: y,
        x_covariates: x,
        phenotype_names: pheno_cols,
    })
}

pub fn mat_to_polars(mat: Mat<f32>, column_names: &[String]) -> Result<DataFrame> {
    if mat.ncols() != column_names.len() {
        bail!("Matrix and column names must have the same length");
    }
    let columns = mat
        .col_iter()
        .zip(column_names)
        .map(|(x, y)| {
            Ok(Column::new(
                y.into(),
                x.iter().copied().collect::<Vec<f32>>(),
            ))
        })
        .collect::<Result<Vec<Column>>>()?;
    let df = DataFrame::new(columns)?;
    Ok(df)
}

/// Build a polars dataframe from a vector of vectors (rows)
pub fn vec_vec_to_polars(vecs: Vec<Vec<f32>>, column_names: &[String]) -> Result<DataFrame> {
    if vecs[0].len() != column_names.len() {
        bail!("Vectors and column names must have the same length");
    }
    let columns = transpose_vec_vec(vecs)
        .iter()
        .zip(column_names)
        .map(|(x, y)| Ok(Column::new(y.into(), x.to_vec())))
        .collect::<Result<Vec<Column>>>()?;
    let df = DataFrame::new(columns)?;
    Ok(df)
}

pub fn vec_vec_to_mat(vecs: Vec<Vec<f32>>) -> Mat<f32> {
    let mut mat = Mat::zeros(vecs.len(), vecs[0].len());
    for (i, row) in vecs.iter().enumerate() {
        for (j, x) in row.iter().enumerate() {
            mat[(i, j)] = *x;
        }
    }
    mat
}

pub fn write_parquet(df: &mut DataFrame, path: &Path) -> Result<()> {
    let file = File::create(path)?;
    let writer = BufWriter::new(file);
    ParquetWriter::new(writer).finish(df)?;
    Ok(())
}

pub fn estimate_genotype_variance(
    phenotype_variance: Expr,
    degrees_of_freedom: Expr,
    beta: Expr,
    std_error: Expr,
) -> Expr {
    phenotype_variance / (degrees_of_freedom * std_error.pow(2) + beta.pow(2))
}
