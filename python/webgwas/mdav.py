import numpy as np
from numpy.typing import NDArray

from webgwas._lowlevel import mdav_impl


def mdav(records: NDArray[np.float64], k: int) -> NDArray[np.float64]:
    """Anonymize a set of records using the MDAV algorithm.

    MDAV ensures that every record is at least k-anonymous, meaning that
    there are at least k - 1 records with the same feature values. MDAV
    anonymizes the records by forming groups of k records and assigning
    each group the mean feature values of the group.

    Args:
        records: A 2D array of records (samples by features)
        k: The anonymization parameter

    Returns:
        A 2D array of records (samples by features)
    """
    records = np.asarray(records)
    assert isinstance(records, np.ndarray), "records must be a numpy array"
    assert records.ndim == 2, "records must be a 2D array"
    assert k > 0, "k must be greater than 0"
    if records.dtype != np.float64:
        records = records.astype(np.float64)
    return mdav_impl(records, k)
