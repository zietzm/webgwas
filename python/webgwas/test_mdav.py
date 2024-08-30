import numpy as np
import pytest

from webgwas._lowlevel import mdav_impl
from webgwas.mdav import mdav


@pytest.mark.parametrize(
    "dtype",
    [
        np.float32,
        np.float64,
    ],
)
def test_mdav_dtype(dtype):
    # Check that mdav works regardless of the dtype of the input (float32 or float64)
    records = np.array(
        [
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        ],
        dtype=dtype,
    )
    # For this test, k=3 with N=3 means they should all be the same (the mean)
    expected = np.vstack(records.mean(axis=0)).reshape(1, -1).tolist()
    assert np.allclose(mdav(records.tolist(), 3), expected)


def test_mdav_numpy():
    records = np.array(
        [
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        ],
    )
    # For this test, k=3 with N=3 means they should all be the same (the mean)
    expected = np.vstack(records.mean(axis=0)).reshape(1, -1).tolist()
    assert np.allclose(mdav_impl(records, 3), expected)
