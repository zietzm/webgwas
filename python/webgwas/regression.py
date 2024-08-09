import numpy as np
import pandas as pd
from numpy.typing import NDArray


def regress_array(y: NDArray, x: NDArray) -> NDArray:
    """Regress y on x.

    Args:
        y: The dependent variable(s)
        x: The independent variable(s)

    Returns:
        The regression coefficients
    """
    beta, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    return beta


def regress(y: pd.Series, x: pd.DataFrame) -> pd.Series:
    """Regress y on x.

    Args:
        y: The dependent variable
        x: The independent variable(s)

    Returns:
        The regression coefficients
    """
    return pd.Series(regress_array(np.asarray(y.values), x.values), index=x.columns)


def regress_left_inverse(y: pd.Series, left_inverse: pd.DataFrame) -> pd.Series:
    """Regress y on x using the left inverse of x.

    Args:
        y: The dependent variable (N x 1)
        left_inverse: The left inverse of x (m x N)

    Returns:
        The regression coefficients
    """
    return pd.Series(left_inverse @ y, index=left_inverse.index)


def residualize(Y: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    """Residualize Y on X.

    Args:
        Y: The dependent variable(s) (N x p)
        X: The independent variable(s) (N x m)

    Returns:
        The residualized Y (N x p)
    """
    beta_array = regress_array(Y.values, X.values)
    beta_df = pd.DataFrame(beta_array, index=X.columns, columns=Y.columns)
    return Y - X @ beta_df


def compute_left_inverse(X: pd.DataFrame) -> pd.DataFrame:
    """Compute the left inverse of X -> inv(X' X) X'

    Args:
        X: The independent variable(s) (N x m)

    Returns:
        The left inverse of X (m x N)
    """
    left_inverse = np.linalg.inv(X.T @ X) @ X.T
    return pd.DataFrame(left_inverse, index=X.columns, columns=X.index)
