"""Tests for apantias.compute.stats."""

# Zarr's stubs report open_array as partially unknown in strict mode; this is a
# test file, so we relax that check rather than cluttering each assertion.
# pyright: reportUnknownMemberType=false

from pathlib import Path

import dask.array as da
import numpy as np
import numpy.typing as npt
import zarr

from apantias.compute.stats import (
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
    infers the parameter as ``str`` because the default is ``"auto"``.
    """
    return da.from_array(data, chunks=chunks)  # type: ignore[reportArgumentType]


def test_compute_median(tmp_path: Path) -> None:
    data: npt.NDArray[np.float64] = np.arange(4 * 2 * 3 * 2, dtype=np.float64).reshape(4, 2, 3, 2)
    d = _from_array(data, chunks=(4, 2, 3, 2))
    out = tmp_path / "median.zarr"
    compute_median(d).to_zarr(out, overwrite=True)  # type: ignore[reportUnknownMemberType]
    result = _read_zarr(out)
    expected = np.median(data, axis=(0, 2))
    np.testing.assert_allclose(result, expected)


def test_compute_offset_corr(tmp_path: Path) -> None:
    data: npt.NDArray[np.float64] = np.ones((4, 2, 3, 2), dtype=np.float64)
    offset: npt.NDArray[np.float64] = np.ones((2, 2), dtype=np.float64) * 0.5
    d = _from_array(data, chunks=(4, 2, 3, 2))
    o = _from_array(offset, chunks=(2, 2))
    out = tmp_path / "offset_corr.zarr"
    compute_offset_corr(d, o).to_zarr(out, overwrite=True)  # type: ignore[reportUnknownMemberType]
    result = _read_zarr(out)
    expected = data - offset[None, :, None, :]
    np.testing.assert_allclose(result, expected)


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
