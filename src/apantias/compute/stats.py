"""Statistical reductions used by the standard analysis."""

from pathlib import Path
from typing import Any, cast

import dask.array as da
import numpy as np


def compute_median(data_p: da.Array, path: str | Path) -> None:
    median_array = da.median(data_p, axis=(0, 2))
    # rechunk to a single chunk and write to zarr
    median_array.rechunk(-1).to_zarr(path)


def compute_offset_corr(data_f: da.Array, median: da.Array, path: str | Path) -> None:
    # Rechunk axis 1 (columns) to a single block so downstream output drops the
    # inner column chunks and processes full frames.
    data_f = data_f.rechunk(cast(Any, {1: -1}))
    offset_corr_array = data_f - median[np.newaxis, :, np.newaxis, :]
    offset_corr_array.to_zarr(path)


def compute_common_modes(data: da.Array, path: str | Path) -> None:
    common_modes_array = da.median(data, axis=3)
    common_modes_array.to_zarr(path)


def compute_slopes(data: da.Array, path: str | Path) -> None:
    # shape is (frames, columns, nreps, rows)
    n = data.shape[2]  # nreps is axis 2
    x = np.arange(n, dtype=np.float64)  # plain NumPy, tiny
    x_dev = x - x.mean()  # plain NumPy
    denominator = float((x_dev**2).sum())  # scalar, computed now

    # Multiply along axis 2 (nreps), then sum along axis 2
    # x_dev shape must broadcast to (1, 1, nreps, 1)
    slopes_array = (data * x_dev[np.newaxis, np.newaxis, :, np.newaxis]).sum(axis=2) / denominator
    slopes_array.to_zarr(path)


def subtract(data: da.Array, common_modes: da.Array, path: str | Path) -> None:
    signals_array = data - common_modes[:, :, :, np.newaxis]
    signals_array.to_zarr(path)


def compute_msd(data: da.Array, path: str | Path) -> None:
    """
    Computes the Mean Squared Deviation for each pixel.

    The data must already be offset (and common mode) corrected, so the deviation is taken from zero.
    """
    # 1. Calculate squared deviations
    squared_diff = data**2

    # 2. Average over frames (axis 0) and repetitions (axis 2)
    msd_array = da.mean(squared_diff, axis=(0, 2))

    # 3. Store the result
    # We rechunk to -1 because the output is tiny (64x64)
    msd_array.rechunk(-1).to_zarr(path)


def compute_signals_mean(signals: da.Array, path: str | Path) -> None:
    signals_median_array = da.mean(signals, axis=2)
    signals_median_array.to_zarr(path)
