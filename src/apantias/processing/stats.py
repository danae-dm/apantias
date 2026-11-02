# dask.array statistical methods are not fully typed by its stubs; the runtime
# contract is a dask Array, so strict unknown-type checks are relaxed here.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

"""Statistical reductions used by the standard analysis."""

import dask.array as da
import numpy as np

from .axis import FRAME, REP, ROW


def compute_median(data_p: da.Array) -> da.Array:
    median_array: da.Array = da.median(data_p, axis=(FRAME, REP))  # type: ignore[reportUnknownMemberType]
    return median_array.rechunk(-1)  # type: ignore[reportUnknownMemberType]


def compute_offset_corr(data_f: da.Array, offset: da.Array, median: da.Array) -> da.Array:
    # Both offset and median have shape (n_cols, n_rows) and broadcast into
    # the 4D frame array as (1, n_cols, 1, n_rows).
    return data_f - offset[None, :, None, :] - median[None, :, None, :]  # type: ignore[reportUnknownVariableType]


def compute_common_modes(data: da.Array) -> da.Array:
    return da.median(data, axis=ROW)  # type: ignore[reportUnknownMemberType]


def compute_slopes(data: da.Array) -> da.Array:
    # shape is (frames, columns, nreps, rows)
    n = data.shape[REP]  # nreps
    x = np.arange(n, dtype=np.float64)  # plain NumPy, tiny
    x_dev = x - x.mean()  # plain NumPy
    denominator = float((x_dev**2).sum())  # scalar, computed now

    # Multiply along the rep axis, then sum along it;
    # x_dev shape must broadcast to (1, 1, nreps, 1).
    slopes_array: da.Array = (data * x_dev[None, None, :, None]).sum(axis=REP) / denominator  # type: ignore[reportUnknownMemberType]
    return slopes_array


def subtract(data: da.Array, common_modes: da.Array) -> da.Array:
    # common_modes has shape (n_frames, n_cols, n_reps); the trailing None
    # restores the row axis so the subtraction broadcasts per pixel.
    return data - common_modes[:, :, :, None]  # type: ignore[reportUnknownVariableType]


def compute_msd(data: da.Array) -> da.Array:
    squared_diff = data**2  # type: ignore[reportUnknownVariableType]
    msd_array: da.Array = da.mean(squared_diff, axis=(FRAME, REP))  # type: ignore[reportUnknownMemberType]
    # Output is tiny (n_cols, n_rows); one chunk is fine.
    return msd_array.rechunk(-1)  # type: ignore[reportUnknownMemberType]


def compute_signals_mean(signals: da.Array) -> da.Array:
    return da.mean(signals, axis=REP)  # type: ignore[reportUnknownMemberType]
