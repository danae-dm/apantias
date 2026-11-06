# dask.array statistical methods are not fully typed by its stubs; the runtime
# contract is a dask Array, so strict unknown-type checks are relaxed here.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

"""Statistical reductions used by the standard analysis."""

import dask.array as da
import numpy as np

from .axis import FRAME, REP, ROW

_MAD_SCALE = 1.4826  # scales MAD to approximate std for normal data


def compute_median(data_p: da.Array) -> da.Array:
    """Compute pixel-wise median across frames.

    Args:
        data_p: Pixel-chunked data with shape (n_frames, n_rows, n_reps, n_cols).

    Returns:
        Median per pixel and repetition with shape (n_rows, n_reps, n_cols).
    """
    median_array: da.Array = da.median(data_p, axis=(FRAME))  # type: ignore[reportUnknownMemberType]
    return median_array.rechunk(-1)  # type: ignore[reportUnknownMemberType]


def compute_offset_corr(data_f: da.Array, offset: da.Array) -> da.Array:
    """Subtract an offset array from frame data.

    Args:
        data_f: Frame-chunked data with shape (n_frames, n_rows, n_reps, n_cols).
        offset: Offset array with shape (n_rows, n_cols) or (n_rows, n_reps, n_cols).

    Returns:
        Corrected data with the same shape as ``data_f``.

    The offset is broadcast over the frame axis, and over the repetition axis
    when a 2D (row, col) offset is supplied.
    """
    _, n_rows, n_reps, n_cols = data_f.shape
    if offset.ndim == 2:
        if offset.shape != (n_rows, n_cols):
            raise ValueError(
                f"2D offset shape {offset.shape} does not match the data layout "
                f"(n_rows, n_cols) = ({n_rows}, {n_cols})."
            )
        return data_f - offset[None, :, None, :]  # type: ignore[reportUnknownVariableType]
    if offset.ndim == 3:
        if offset.shape != (n_rows, n_reps, n_cols):
            raise ValueError(
                f"3D offset shape {offset.shape} does not match the data layout "
                f"(n_rows, n_reps, n_cols) = ({n_rows}, {n_reps}, {n_cols})."
            )
        return data_f - offset[None, ...]  # type: ignore[reportUnknownVariableType]
    raise ValueError(
        f"Offset must be 2D (n_rows, n_cols) or 3D (n_rows, n_reps, n_cols), "
        f"got {offset.ndim}D array with shape {offset.shape}."
    )


def compute_common_modes(data: da.Array) -> da.Array:
    """Compute common-mode signal per frame, row and repetition.

    Args:
        data: Frame-chunked data with shape (n_frames, n_rows, n_reps, n_cols).

    Returns:
        Common-mode signal with shape (n_frames, n_reps, n_cols).
        This is the median across the row axis (axis 1).
    """
    return da.median(data, axis=ROW)  # type: ignore[reportUnknownMemberType]


def compute_slopes(data: da.Array) -> da.Array:
    """Compute linear slope of signal versus repetition index.

    Args:
        data: Corrected signals with shape (n_frames, n_rows, n_reps, n_cols).

    Returns:
        Slopes per pixel with shape (n_frames, n_rows, n_cols).
        Computed by fitting a line to the repetition axis (axis 2).
    """
    # shape is (n_frames, n_rows, n_reps, n_cols)
    n = data.shape[REP]  # nreps
    x = np.arange(n, dtype=np.float64)  # plain NumPy, tiny
    x_dev = x - x.mean()  # plain NumPy
    denominator = float((x_dev**2).sum())  # scalar, computed now

    # Multiply along the rep axis, then sum along it;
    # x_dev shape must broadcast to (1, 1, nreps, 1).
    slopes_array: da.Array = (data * x_dev[None, None, :, None]).sum(axis=REP) / denominator  # type: ignore[reportUnknownMemberType]
    return slopes_array


def subtract(data: da.Array, common_modes: da.Array) -> da.Array:
    """Subtract common-mode signal from corrected data.

    Args:
        data: Offset-corrected data with shape (n_frames, n_rows, n_reps, n_cols).
        common_modes: Common-mode signal with shape (n_frames, n_reps, n_cols).

    Returns:
        Common-mode corrected data with same shape as input.
    """
    # common_modes has shape (n_frames, n_reps, n_cols); the trailing None
    # restores the row axis so the subtraction broadcasts per pixel.
    return data - common_modes[:, :, :, None]  # type: ignore[reportUnknownVariableType]


def compute_msd(data: da.Array) -> da.Array:
    """Compute mean squared deviation of the corrected signal.

    Args:
        data: Corrected signals with shape (n_frames, n_rows, n_reps, n_cols).

    Returns:
        MSD per pixel with shape (n_rows, n_cols).
        Computed as mean of squared values across frame and rep axes.
    """
    squared_diff = data**2  # type: ignore[reportUnknownVariableType]
    msd_array: da.Array = da.mean(squared_diff, axis=(FRAME, REP))  # type: ignore[reportUnknownMemberType]
    # Output is tiny (n_rows, n_cols); one chunk is fine.
    return msd_array.rechunk(-1)  # type: ignore[reportUnknownMemberType]


def compute_signals_mean(signals: da.Array) -> da.Array:
    """Compute mean corrected signal per frame and pixel.

    Args:
        signals: Common-mode corrected signals with shape
                 (n_frames, n_rows, n_reps, n_cols).

    Returns:
        Mean signals with shape (n_frames, n_rows, n_cols).
        Computed by averaging across the rep axis (axis 2).
    """
    return da.mean(signals, axis=REP)  # type: ignore[reportUnknownMemberType]


def mask_bad_slopes(
    slopes: da.Array,
    signals: da.Array,
    slope_threshold: float,
) -> da.Array:
    """Mask signal repetitions whose pixel slopes are outliers.

    Computes the median slope and the scaled median absolute deviation
    (MAD * 1.4826) across all pixels and frames. Pixels whose slope
    deviates by more than ``slope_threshold * scaled_MAD`` from the
    median are marked as bad, and all repetitions of those pixels in
    ``signals`` are replaced with ``np.nan``.

    If ``slope_threshold`` is 0, ``signals`` is returned unchanged.

    Args:
        slopes: Linear slopes with shape (n_frames, n_rows, n_cols).
        signals: Corrected signals with shape
                 (n_frames, n_rows, n_reps, n_cols).
        slope_threshold: Multiplier for the scaled-MAD outlier threshold.

    Returns:
        Copy of ``signals`` with outlier pixels masked by ``np.nan``.

    Raises:
        ValueError: If ``slopes`` contains NaN values.
    """
    if slope_threshold == 0:
        return signals

    # Slopes is small (frames * rows * cols). Compute median/MAD eagerly to
    # avoid expensive Dask P2P shuffle/rechunk operations from da.median on a
    # flattened array. The actual masking of the full signal stays lazy.
    slopes_np = slopes.compute()  # type: ignore[reportUnknownVariableType]
    if np.isnan(slopes_np).any():
        raise ValueError("no NaN values should be present in slopes at this stage")

    center = float(np.median(slopes_np))
    scaled_mad = float(np.median(np.abs(slopes_np - center))) * _MAD_SCALE

    outliers = da.abs(slopes - center) > slope_threshold * scaled_mad  # type: ignore[reportUnknownVariableType]
    # Align the outlier mask with the chunking of the full signal array before
    # broadcasting over the repetition axis. This avoids an implicit Dask
    # rechunk/P2P shuffle when the stored slopes have a different chunk layout
    # than signals_corr.
    outliers = outliers.rechunk((signals.chunks[0], signals.chunks[1], signals.chunks[3]))  # type: ignore[reportUnknownVariableType]
    outliers = outliers[:, :, None, :]

    return da.where(outliers, np.nan, signals)  # type: ignore[reportUnknownVariableType]
