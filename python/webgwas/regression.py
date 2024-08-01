import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame, Series


def regress_array(y: NDArray, x: NDArray) -> NDArray:
    """Regress y on x.

    Args:
        y: The dependent variable
        x: The independent variable(s)

    Returns:
        The regression coefficients
    """
    beta, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    return beta


def regress(y: Series, x: DataFrame) -> Series:
    """Regress y on x.

    Args:
        y: The dependent variable
        x: The independent variable(s)

    Returns:
        The regression coefficients
    """
    return Series(regress_array(np.asarray(y.values), x.values), index=x.columns)


def regress_left_inverse(y: Series, left_inverse: DataFrame) -> Series:
    """Regress y on x using the left inverse of x.

    Args:
        y: The dependent variable (N x 1)
        left_inverse: The left inverse of x (m x N)

    Returns:
        The regression coefficients
    """
    return Series(left_inverse @ y, index=left_inverse.index)
