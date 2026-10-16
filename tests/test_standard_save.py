"""Tests for the final-output writing of the standard analysis."""

# Tests import package-internal helpers and exercise Dask/Zarr's runtime API,
# which has incomplete stubs in strict mode.
# pyright: reportPrivateUsage=false, reportUnknownMemberType=false, reportUnknownVariableType=false

import dask.array as da
import numpy as np
import numpy.typing as npt

from apantias.standard import StandardAnalysis


def test_open_zarr_output_and_batch_store(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Regression: da.store's batch form needs zarr array targets, not paths.

    Passing PosixPath targets made dask fail with
    "'PosixPath' object does not support item assignment" on the workers.
    """
    a: npt.NDArray[np.float64] = np.arange(24, dtype=np.float64).reshape(4, 6)
    b: npt.NDArray[np.float64] = np.arange(24, dtype=np.float64).reshape(4, 6) * 2
    sources = [da.from_array(a, chunks=(2, 3)), da.from_array(b, chunks=(2, 3))]

    targets = [
        StandardAnalysis._open_zarr_output(tmp_path / name, arr)
        for name, arr in [("out_a", sources[0]), ("out_b", sources[1])]
    ]
    da.store(sources, targets, lock=False)

    np.testing.assert_allclose(da.from_zarr(tmp_path / "out_a").compute(), a)
    np.testing.assert_allclose(da.from_zarr(tmp_path / "out_b").compute(), b)


def test_open_zarr_output_overwrites(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Re-running must replace a previous output array instead of failing."""
    path = tmp_path / "out"
    arr = da.zeros((4, 6), chunks=(2, 3))
    da.store(arr, StandardAnalysis._open_zarr_output(path, arr), lock=False)

    arr2 = da.ones((4, 6), chunks=(2, 3))
    da.store(arr2, StandardAnalysis._open_zarr_output(path, arr2), lock=False)

    np.testing.assert_allclose(da.from_zarr(path).compute(), np.ones((4, 6)))
