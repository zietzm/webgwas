import pandas as pd
import polars as pl

from webgwas._lowlevel import Projection, run_igwas, run_igwas_df


def estimate_genotype_variance(
    phenotype_variance: float,
    degrees_of_freedom: int,
    std_error: float,
    beta: float,
) -> float:
    return phenotype_variance / (degrees_of_freedom * std_error**2 + beta**2)


def igwas_prod(
    projection_vector: pd.Series,
    covariance_matrix: pd.DataFrame,
    gwas_result_path: str,
    output_file_path: str,
    num_covar: int = 1,
):
    projection_variance = projection_vector @ covariance_matrix @ projection_vector
    projection = Projection(
        feature_id=projection_vector.index,
        feature_coefficient=projection_vector,
    )
    run_igwas(
        projection=projection,
        projection_variance=projection_variance,
        n_covariates=num_covar,
        input_path=gwas_result_path,
        output_path=output_file_path,
    )


def igwas_prod_df(
    projection_vector: pd.Series,
    covariance_matrix: pd.DataFrame,
    gwas_result_df: pl.DataFrame,
    output_file_path: str,
    num_covar: int = 1,
):
    projection_variance = projection_vector @ covariance_matrix @ projection_vector
    projection = Projection(
        feature_id=projection_vector.index,
        feature_coefficient=projection_vector,
    )
    run_igwas_df(
        gwas_df=gwas_result_df,
        projection=projection,
        projection_variance=projection_variance,
        n_covariates=num_covar,
        output_path=output_file_path,
    )
