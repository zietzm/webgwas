import os
import pathlib
import tempfile

import numpy as np
import pandas as pd
import polars as pl
import pytest

from webgwas.igwas import estimate_genotype_variance, igwas, igwas_files, igwas_prod


@pytest.fixture
def test_data():
    return {
        "n_samples": 100,
        "n_variants": 100,
        "n_covariates": 10,
        "n_phenotypes": 10,
        "n_projections": 10,
    }


@pytest.fixture(scope="module")
def temp_dir():
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture
def setup_test_data(test_data, temp_dir):
    def _setup(compress=False) -> list[str]:
        # Generate test data
        np.random.seed(42)
        genotypes = np.random.randint(
            0, 3, size=(test_data["n_samples"], test_data["n_variants"])
        ).astype(np.float32)
        phenotypes = np.random.randn(
            test_data["n_samples"], test_data["n_phenotypes"]
        ).astype(np.float32)
        covariates = np.random.randn(
            test_data["n_samples"], test_data["n_covariates"]
        ).astype(np.float32)
        projection_matrix = np.random.normal(
            size=(test_data["n_phenotypes"], test_data["n_projections"])
        ).astype(np.float32)

        # Compute covariance matrix
        phenotype_residuals = (
            phenotypes
            - covariates @ np.linalg.lstsq(covariates, phenotypes, rcond=None)[0]
        )
        covariance_matrix = np.cov(phenotype_residuals.T)

        # Write files
        phenotype_names = [f"feat_{i}" for i in range(test_data["n_phenotypes"])]
        projection_names = [f"proj_{i}" for i in range(test_data["n_projections"])]
        phenotype_idx = pd.Index(phenotype_names, name="phenotype_id")
        projection_idx = pd.Index(projection_names, name="projection_id")
        pd.DataFrame(
            projection_matrix, columns=projection_idx, index=phenotype_idx
        ).to_csv(
            os.path.join(temp_dir, "projection_matrix.csv"),
        )
        pd.DataFrame(
            covariance_matrix, index=phenotype_idx, columns=phenotype_idx
        ).to_csv(
            os.path.join(temp_dir, "covariance_matrix.csv"),
        )

        # Generate GWAS results
        gwas_results = []
        for i in range(test_data["n_phenotypes"]):
            phenotype = phenotypes[:, i]
            results = []
            for j in range(test_data["n_variants"]):
                genotype = genotypes[:, j]
                beta, std_error = perform_direct_gwas(genotype, phenotype, covariates)
                results.append(
                    {
                        "variant_id": f"variant_{j}",
                        "beta": beta,
                        "std_error": std_error,
                        "sample_size": test_data["n_samples"],
                    }
                )

            file_name = f"feat_{i}.tsv" + (".zst" if compress else "")
            file_path = os.path.join(temp_dir, file_name)
            pd.DataFrame(results).to_csv(
                file_path,
                index=False,
                compression="zstd" if compress else None,
                sep="\t",
            )
            gwas_results.append(file_path)

        return gwas_results

    return _setup


def perform_direct_gwas(genotype, phenotype, covariates):
    X = np.column_stack((np.ones(len(genotype)), covariates, genotype))
    beta = np.linalg.lstsq(X, phenotype, rcond=None)[0]
    residuals = phenotype - X @ beta
    sigma2 = np.sum(residuals**2) / (len(phenotype) - X.shape[1])
    var_beta = sigma2 * np.linalg.inv(X.T @ X)[-1, -1]
    std_error = np.sqrt(var_beta)
    return beta[-1], std_error


@pytest.mark.parametrize("compress", [False, True])
def test_igwas_files(setup_test_data, test_data, temp_dir, compress):
    gwas_results = setup_test_data(compress)

    suffix = ".csv.zst" if compress else ".csv"
    output_file = os.path.join(temp_dir, f"igwas_results{suffix}")

    assert os.path.exists(os.path.join(temp_dir, "projection_matrix.csv"))
    assert os.path.exists(os.path.join(temp_dir, "covariance_matrix.csv"))

    for result in gwas_results:
        assert os.path.exists(result)

    igwas_files(
        projection_matrix_path=os.path.join(temp_dir, "projection_matrix.csv"),
        covariance_matrix_path=os.path.join(temp_dir, "covariance_matrix.csv"),
        gwas_result_paths=gwas_results,
        output_file_path=output_file,
        num_covar=test_data["n_covariates"],
        chunksize=test_data["n_variants"],
        variant_id="variant_id",
        beta="beta",
        std_error="std_error",
        sample_size="sample_size",
        num_threads=2,
        capacity=10,
        compress=compress,
        quiet=True,
    )

    results = pd.read_csv(output_file, sep="\t")
    assert results.shape[0] == test_data["n_variants"] * test_data["n_projections"]
    assert set(results.columns) == {
        "variant_id",
        "beta",
        "std_error",
        "t_stat",
        "neg_log_p_value",
        "sample_size",
    }


@pytest.mark.parametrize("compress", [False, True])
def test_igwas(setup_test_data, test_data, temp_dir, compress):
    gwas_results = setup_test_data(compress)

    suffix = ".csv.zst" if compress else ".csv"
    output_file = os.path.join(temp_dir, f"igwas_results{suffix}")

    assert os.path.exists(os.path.join(temp_dir, "projection_matrix.csv"))
    assert os.path.exists(os.path.join(temp_dir, "covariance_matrix.csv"))

    for result in gwas_results:
        assert os.path.exists(result)

    projection_matrix = pd.read_csv(
        os.path.join(temp_dir, "projection_matrix.csv"), index_col=0
    )
    covariance_matrix = pd.read_csv(
        os.path.join(temp_dir, "covariance_matrix.csv"), index_col=0
    )
    igwas(
        projection_matrix=projection_matrix,
        covariance_matrix=covariance_matrix,
        gwas_result_paths=gwas_results,
        output_file_path=output_file,
        num_covar=test_data["n_covariates"],
        chunksize=test_data["n_variants"],
        variant_id="variant_id",
        beta="beta",
        std_error="std_error",
        sample_size="sample_size",
        num_threads=2,
        capacity=10,
        compress=compress,
        quiet=True,
    )


@pytest.mark.parametrize("compress", [False, True])
def test_igwas_prod(setup_test_data, test_data, temp_dir, compress):
    gwas_results = setup_test_data(compress)
    suffix = ".csv.zst" if compress else ".csv"
    covariance_matrix = pd.read_csv(
        os.path.join(temp_dir, "covariance_matrix.csv"), index_col=0
    )
    projection_matrix = (
        pd.read_csv(os.path.join(temp_dir, "projection_matrix.csv"), index_col=0)
        .iloc[:, 0]
        .pipe(lambda s: s / np.sqrt(s @ covariance_matrix @ s))
        .to_frame()
    )
    assert projection_matrix.iloc[:, 0] @ covariance_matrix @ projection_matrix.iloc[
        :, 0
    ] == pytest.approx(1.0, rel=1e-4)

    schema = {
        "variant_id": pl.Utf8,
        "beta": pl.Float32,
        "std_error": pl.Float32,
        "sample_size": pl.Int32,
    }
    full_gwas_df = None
    for gwas_path in gwas_results:
        gwas_name = pathlib.Path(gwas_path).name.replace(".tsv", "").replace(".zst", "")
        gwas_df = (
            pl.read_csv(gwas_path, separator="\t", schema=schema)
            .with_columns(
                degrees_of_freedom=(
                    pl.col("sample_size") - test_data["n_covariates"] - 2
                ).cast(pl.Int32),
            )
            .with_columns(
                genotype_variance=estimate_genotype_variance(
                    phenotype_variance=covariance_matrix.loc[gwas_name, gwas_name],
                    degrees_of_freedom=pl.col("degrees_of_freedom"),
                    std_error=pl.col("std_error"),
                    beta=pl.col("beta"),
                ).cast(pl.Float32),
            )
            .select(
                "variant_id",
                pl.struct(["beta", "degrees_of_freedom", "genotype_variance"]).alias(
                    gwas_name
                ),
            )
        )
        if full_gwas_df is None:
            full_gwas_df = gwas_df
        else:
            full_gwas_df = full_gwas_df.join(gwas_df, on=["variant_id"])

    assert isinstance(full_gwas_df, pl.DataFrame)
    full_gwas_df.write_parquet(os.path.join(temp_dir, "full_gwas.parquet"))

    # Run using the production API
    prod_output_file = os.path.join(temp_dir, f"igwas_prod_results{suffix}")
    igwas_prod(
        projection_vector=projection_matrix.iloc[:, 0],
        covariance_matrix=covariance_matrix,
        gwas_result_path=os.path.join(temp_dir, "full_gwas.parquet"),
        output_file_path=prod_output_file,
        num_covar=test_data["n_covariates"],
    )

    # Run using the file API
    files_output_file = os.path.join(temp_dir, f"igwas_files_results{suffix}")
    igwas(
        projection_matrix=projection_matrix.iloc[:, [0]],
        covariance_matrix=covariance_matrix,
        gwas_result_paths=gwas_results,
        output_file_path=files_output_file,
        num_covar=test_data["n_covariates"],
        chunksize=test_data["n_variants"],
        variant_id="variant_id",
        beta="beta",
        std_error="std_error",
        sample_size="sample_size",
        num_threads=1,
        capacity=1,
        compress=compress,
        quiet=True,
    )

    # Check that the results are the same
    prod_result_df = pl.read_csv(prod_output_file, separator="\t")
    files_result_df = pl.read_csv(files_output_file, separator="\t")
    merged_result_df = prod_result_df.join(
        files_result_df, on=["variant_id"], suffix="_files"
    )
    for col in ["beta", "std_error", "t_stat", "neg_log_p_value", "sample_size"]:
        max_diff = (
            (merged_result_df[col] - merged_result_df[col + "_files"]).abs().max()
        )
        assert max_diff == pytest.approx(
            0.0, abs=1e-6
        ), f"Max diff for {col}: {max_diff}"
