# dask.array statistical methods are not fully typed by their stubs; the runtime
# contract is a dask Array, so strict unknown-type checks are relaxed here.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

"""Statistical reductions used by the standard analysis."""

from pathlib import Path

import dask.array as da
import numpy as np


def compute_median(data_p: da.Array, path: str | Path) -> None:
    # dask's median/mean/rechunk/to_zarr type stubs are incomplete in strict
    # mode; the runtime contract is a dask Array, so we ignore the unknown types.
    median_array: da.Array = da.median(data_p, axis=(0, 2))  # type: ignore[reportUnknownMemberType]
    median_array.rechunk(-1).to_zarr(path)  # type: ignore[reportUnknownMemberType]


def compute_offset_corr(data_f: da.Array, median: da.Array, path: str | Path) -> None:
    data_f = data_f.rechunk({1: -1})  # type: ignore[reportUnknownMemberType]
    offset_corr_array = data_f - median[np.newaxis, :, np.newaxis, :]  # type: ignore[reportUnknownVariableType]
    offset_corr_array.to_zarr(path)  # type: ignore[reportUnknownMemberType]


def compute_common_modes(data: da.Array, path: str | Path) -> None:
    common_modes_array: da.Array = da.median(data, axis=3)  # type: ignore[reportUnknownMemberType]
    common_modes_array.to_zarr(path)  # type: ignore[reportUnknownMemberType]


def compute_slopes(data: da.Array, path: str | Path) -> None:
    # shape is (frames, columns, nreps, rows)
    n = data.shape[2]  # nreps is axis 2
    x = np.arange(n, dtype=np.float64)  # plain NumPy, tiny
    x_dev = x - x.mean()  # plain NumPy
    denominator = float((x_dev**2).sum())  # scalar, computed now

    # Multiply along axis 2 (nreps), then sum along axis 2
    # x_dev shape must broadcast to (1, 1, nreps, 1)
    slopes_array: da.Array = (data * x_dev[np.newaxis, np.newaxis, :, np.newaxis]).sum(axis=2) / denominator
    slopes_array.to_zarr(path)  # type: ignore[reportUnknownMemberType]


def subtract(data: da.Array, common_modes: da.Array, path: str | Path) -> None:
    signals_array = data - common_modes[:, :, :, np.newaxis]  # type: ignore[reportUnknownVariableType]
    signals_array.to_zarr(path)  # type: ignore[reportUnknownMemberType]


def compute_msd(data: da.Array, path: str | Path) -> None:
    """
    Computes the Mean Squared Deviation for each pixel.

    The data must already be offset (and common mode) corrected, so the deviation is taken from zero.
    """
    # 1. Calculate squared deviations
    squared_diff = data**2  # type: ignore[reportUnknownVariableType]

    # 2. Average over frames (axis 0) and repetitions (axis 2)
    msd_array: da.Array = da.mean(squared_diff, axis=(0, 2))  # type: ignore[reportUnknownMemberType]

    # 3. Store the result
    # We rechunk to -1 because the output is tiny (64x64)
    msd_array.rechunk(-1).to_zarr(path)  # type: ignore[reportUnknownMemberType]


def compute_signals_mean(signals: da.Array, path: str | Path) -> None:
    signals_median_array: da.Array = da.mean(signals, axis=2)  # type: ignore[reportUnknownMemberType]
    signals_median_array.to_zarr(path)  # type: ignore[reportUnknownMemberType]
