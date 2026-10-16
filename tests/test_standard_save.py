"""Tests for the final-output writing of the standard analysis."""

# Tests import package-internal helpers and exercise Dask/Zarr's runtime API,
# which has incomplete stubs in strict mode (e.g. from_array's chunks parameter
# is stubbed as `str` only). Relaxed per this module's header.
# pyright: reportPrivateUsage=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from pathlib import Path
from typing import cast

import dask.array as da
import numpy as np
import numpy.typing as npt

from apantias.standard import StandardAnalysis


def test_open_zarr_output_and_batch_store(tmp_path: Path) -> None:
    """Regression: da.store's batch form needs zarr array targets, not paths.

    Passing PosixPath targets made dask fail with
    "'PosixPath' object does not support item assignment" on the workers.
    """
    a: npt.NDArray[np.float64] = np.arange(24, dtype=np.float64).reshape(4, 6)
    b: npt.NDArray[np.float64] = np.arange(24, dtype=np.float64).reshape(4, 6) * 2
    sources: list[da.Array] = [
        da.from_array(a, chunks=(2, 3)),  # type: ignore[reportArgumentType]
        da.from_array(b, chunks=(2, 3)),  # type: ignore[reportArgumentType]
    ]

    pairs = [("out_a", sources[0]), ("out_b", sources[1])]
    targets = [StandardAnalysis._open_zarr_output(tmp_path / name, arr) for name, arr in pairs]
    da.store(sources, targets, lock=False)

    result_a = cast(da.Array, da.from_zarr(tmp_path / "out_a")).compute()
    result_b = cast(da.Array, da.from_zarr(tmp_path / "out_b")).compute()
    np.testing.assert_allclose(result_a, a)
    np.testing.assert_allclose(result_b, b)


def test_open_zarr_output_overwrites(tmp_path: Path) -> None:
    """Re-running must replace a previous output array instead of failing."""
    path = tmp_path / "out"
    arr: da.Array = da.zeros((4, 6), chunks=(2, 3))
    da.store(arr, StandardAnalysis._open_zarr_output(path, arr), lock=False)

    arr2: da.Array = da.ones((4, 6), chunks=(2, 3))
    da.store(arr2, StandardAnalysis._open_zarr_output(path, arr2), lock=False)

    result = cast(da.Array, da.from_zarr(path)).compute()
    np.testing.assert_allclose(result, np.ones((4, 6)))


def test_load_offset_falls_back_to_median() -> None:
    """Without ext_offset the applied offset is the own offset (the median)."""
    analysis = StandardAnalysis.__new__(StandardAnalysis)
    analysis.ext_offset = None  # type: ignore[reportAttributeAccessIssue]
    median: da.Array = da.ones((2, 2), chunks=-1)

    assert analysis._load_offset(median) is median
