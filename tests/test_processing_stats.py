"""Tests for apantias.processing.stats."""

# Zarr's stubs report open_array as partially unknown in strict mode; this is a
# test file, so we relax that check rather than cluttering each assertion.
# pyright: reportUnknownMemberType=false

from pathlib import Path
from typing import Any

import dask.array as da
import numpy as np
import numpy.typing as npt
import zarr

from apantias.processing.reductions import apply_pixelwise
from apantias.processing.stats import (
    compute_common_modes,
    compute_median,
    compute_msd,
    compute_offset_corr,
    compute_signals_mean,
    compute_slopes,
    subtract,
)


def _read_zarr(path: Path) -> npt.NDArray[np.float64]:
    """Read a zarr array as a float64 ndarray."""
    arr = zarr.open_array(str(path), mode="r")
    return np.asarray(arr[:], dtype=np.float64)


def _from_array(data: npt.NDArray[np.float64], chunks: tuple[int, ...]) -> da.Array:
    """Wrap da.from_array with a typed chunks parameter.

    dask's runtime signature accepts tuple chunk specifications, but Pyright
    infers the parameter as ``str`` because the default is ``auto".
    """
    return da.from_array(data, chunks=chunks)  # type: ignore[reportArgumentType]


def test_compute_median(tmp_path: Path) -> None:
    data: npt.NDArray[np.float64] = np.arange(4 * 2 * 3 * 2, dtype=np.float64).reshape(4, 2, 3, 2)
    d = _from_array(data, chunks=(4, 2, 3, 2))
    out = tmp_path / "median.zarr"
    compute_median(d).to_zarr(out, overwrite=True)  # type: ignore[reportUnknownMemberType]
    result = _read_zarr(out)
    expected = np.median(data, axis=0)
    np.testing.assert_allclose(result, expected)


def test_compute_offset_corr(tmp_path: Path) -> None:
    """Test offset correction with 2D and 3D offsets (no median subtraction)."""
    data: npt.NDArray[np.float64] = np.ones((4, 2, 3, 2), dtype=np.float64) * 10
    d = _from_array(data, chunks=(4, 2, 3, 2))

    # Test with 2D offset (should broadcast over reps)
    offset_2d: npt.NDArray[np.float64] = np.full((2, 2), 3.0, dtype=np.float64)
    o = _from_array(offset_2d, chunks=(2, 2))
    out = tmp_path / "offset_corr_2d.zarr"
    result_arr = compute_offset_corr(d, o)  # type: ignore[reportUnknownArgumentType]
    result_arr.to_zarr(out, overwrite=True)  # type: ignore[reportUnknownMemberType]
    result = _read_zarr(out)
    expected = data - offset_2d[None, :, None, :]
    np.testing.assert_allclose(result, expected)

    # Test with 3D offset (different offset per rep)
    offset_3d: npt.NDArray[np.float64] = np.full((2, 3, 2), 2.0, dtype=np.float64)
    o3 = _from_array(offset_3d, chunks=(2, 3, 2))
    out3 = tmp_path / "offset_corr_3d.zarr"
    result_arr3 = compute_offset_corr(d, o3)  # type: ignore[reportUnknownArgumentType]
    result_arr3.to_zarr(out3, overwrite=True)  # type: ignore[reportUnknownMemberType]
    result3 = _read_zarr(out3)
    expected3 = data - offset_3d[None, ...]
    np.testing.assert_allclose(result3, expected3)


def test_compute_common_modes(tmp_path: Path) -> None:
    data: npt.NDArray[np.float64] = np.arange(4 * 2 * 3 * 2, dtype=np.float64).reshape(4, 2, 3, 2)
    d = _from_array(data, chunks=(4, 2, 3, 2))
    out = tmp_path / "common_modes.zarr"
    compute_common_modes(d).to_zarr(out, overwrite=True)  # type: ignore[reportUnknownMemberType]
    result = _read_zarr(out)
    expected = np.median(data, axis=3)
    np.testing.assert_allclose(result, expected)


def test_subtract(tmp_path: Path) -> None:
    data: npt.NDArray[np.float64] = np.ones((2, 2, 3, 2), dtype=np.float64) * 4
    common: npt.NDArray[np.float64] = np.ones((2, 2, 3), dtype=np.float64)
    d = _from_array(data, chunks=(2, 2, 3, 2))
    c = _from_array(common, chunks=(2, 2, 3))
    out = tmp_path / "signals.zarr"
    subtract(d, c).to_zarr(out, overwrite=True)  # type: ignore[reportUnknownMemberType]
    result = _read_zarr(out)
    expected = data - common[:, :, :, None]
    np.testing.assert_allclose(result, expected)


def test_compute_msd(tmp_path: Path) -> None:
    data: npt.NDArray[np.float64] = np.ones((4, 2, 3, 2), dtype=np.float64)
    d = _from_array(data, chunks=(4, 2, 3, 2))
    out = tmp_path / "msd.zarr"
    compute_msd(d).to_zarr(out, overwrite=True)  # type: ignore[reportUnknownMemberType]
    result = _read_zarr(out)
    np.testing.assert_allclose(result, 1.0)


def test_compute_signals_mean(tmp_path: Path) -> None:
    data: npt.NDArray[np.float64] = np.arange(4 * 2 * 3 * 2, dtype=np.float64).reshape(4, 2, 3, 2)
    d = _from_array(data, chunks=(4, 2, 3, 2))
    out = tmp_path / "signals_mean.zarr"
    compute_signals_mean(d).to_zarr(out, overwrite=True)  # type: ignore[reportUnknownMemberType]
    result = _read_zarr(out)
    expected = np.mean(data, axis=2)
    np.testing.assert_allclose(result, expected)


def test_compute_slopes(tmp_path: Path) -> None:
    # Data linear in reps: slope should be 1.0 for every pixel.
    nreps = 5
    data: npt.NDArray[np.float64] = np.zeros((2, 2, nreps, 2), dtype=np.float64)
    for i in range(nreps):
        data[:, :, i, :] = i
    d = _from_array(data, chunks=(2, 2, nreps, 2))
    out = tmp_path / "slopes.zarr"
    compute_slopes(d).to_zarr(out, overwrite=True)  # type: ignore[reportUnknownMemberType]
    result = _read_zarr(out)
    np.testing.assert_allclose(result, 1.0, atol=1e-10)


def test_apply_pixelwise() -> None:
    """apply_pixelwise processes each pixel's time series to return parameters.

    Input (n_frames, n_pixels) chunked as (n_frames, 1) produces output
    (n_pixels * k,) where k is the function's output dimension. The output
    can be reshaped to (n_pixels, k) then to spatial dimensions.
    """
    n_frames, n_pixels = 4, 6  # 2x3 grid
    data: npt.NDArray[np.float64] = np.arange(n_frames * n_pixels, dtype=np.float64).reshape(n_frames, n_pixels)
    d = da.from_array(data, chunks=(n_frames, 1))  # type: ignore[reportArgumentType]  # one chunk per pixel

    def func(t: npt.NDArray[Any]) -> npt.NDArray[Any]:
        return np.array([t.max() - t.min(), np.mean(t)])  # 2 parameters per pixel

    result = apply_pixelwise(d, func)  # type: ignore[reportUnknownArgumentType]

    # Result is flattened (n_pixels * k,)
    computed = result.compute()  # type: ignore[assignment]
    assert computed.shape == (n_pixels * 2,)
    expected = np.empty((n_pixels, 2), dtype=np.float64)
    for p in range(n_pixels):
        series = data[:, p]
        expected[p, 0] = series.max() - series.min()
        expected[p, 1] = series.mean()
    # Reshape to compare
    computed_reshaped = computed.reshape(n_pixels, 2)  # type: ignore[union-attr]
    np.testing.assert_allclose(computed_reshaped, expected)  # type: ignore[call-overload]
