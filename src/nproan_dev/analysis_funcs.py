import gc
import os

import numpy as np

from . import parallel_computations
from . import logger

_logger = logger.Logger(__name__, "info").get_logger()


def get_avg_over_frames(data: np.ndarray) -> np.ndarray:
    """
    Calculates the average over the frames in data.
    Args:
        data: np.array (nframes, column_size, nreps, row_size)

    Returns:
        np.array (column_size, nreps, row_size)
    """
    if np.ndim(data) != 4:
        _logger.error("Input data is not a 4D array.")
        raise ValueError("Input data is not a 4D array.")
    return parallel_computations.nanmean(data, axis=0)


def get_avg_over_nreps(data: np.ndarray) -> np.ndarray:
    """
    Calculates the average over the nreps in data.
    Args:
        data: np.array in shape (nframes, column_size, nreps, row_size)
    Returns:
        np.array in shape (nframes, column_size, row_size)
    """
    if np.ndim(data) != 4:
        _logger.error("Input data is not a 4D array.")
        raise ValueError("Input data is not a 4D array.")
    return parallel_computations.nanmean(data, axis=2)


def get_avg_over_frames_and_nreps(
    data: np.ndarray,
    avg_over_frames: np.ndarray = None,
    avg_over_nreps: np.ndarray = None,
) -> np.ndarray:
    """
    Calculates the average over the frames and nreps in data. If avg_over_frames
    or avg_over_nreps are already calculated they can be passed as arguments to
    save computation time.
    Args:
        data: np.array (nframes, column_size, nreps, row_size)
        avg_over_frames: (optional) np.array (column_size, nreps, row_size)
        avg_over_nreps: (optional) np.array (nframes, column_size, row_size)
    Returns:
        np.array (column_size, row_size)
    """
    if np.ndim(data) != 4:
        _logger.error("Input data is not a 4D array.")
        raise ValueError("Input data is not a 4D array.")

    if avg_over_frames is None and avg_over_nreps is None:
        return parallel_computations.nanmean(
            parallel_computations.nanmean(data, axis=0), axis=2
        )

    if avg_over_frames is not None and avg_over_nreps is not None:
        if np.ndim(avg_over_frames) != 3 or np.ndim(avg_over_nreps) != 3:
            _logger.error("Input avg_over_frames or avg_over_nreps is not a 3D array.")
            raise ValueError(
                "Input avg_over_frames or avg_over_nreps is not a 3D array."
            )
        if avg_over_frames.shape[1] < avg_over_nreps.shape[0]:
            return parallel_computations.nanmean(avg_over_frames, axis=1)
        else:
            return parallel_computations.nanmean(avg_over_nreps, axis=0)
    else:
        if avg_over_nreps is not None:
            if np.ndim(avg_over_nreps) != 3:
                _logger.error("Input avg_over_nreps is not a 3D array.")
                raise ValueError("Input avg_over_nreps is not a 3D array.")
            return parallel_computations.nanmean(avg_over_nreps, axis=0)
        else:
            if np.ndim(avg_over_frames) != 3:
                _logger.error("Input avg_over_frames is not a 3D array.")
                raise ValueError("Input avg_over_frames is not a 3D array.")
            return parallel_computations.nanmean(avg_over_frames, axis=1)


def get_rolling_average(data: np.ndarray, window_size: int) -> np.ndarray:
    """
    Calculates a rolling average over window_size
    Args:
        data: 1D np.array
        window_size: int
    Returns:
        1D np.array
    """
    weights = np.repeat(1.0, window_size) / window_size
    # Use 'valid' mode to ensure that output has the same length as input
    return np.convolve(data, weights, mode="valid")


def get_ram_usage_in_gb(
    frames: int, column_size: int, nreps: int, row_size: int
) -> int:
    """
    Calculates the RAM usage in GB for a 4D array of the given dimensions.
    Assuming float64. (8 bytes per element)
    Args:
        frames: int
        column_size: int
        nreps: int
        row_size: int
    Returns:
        estimated RAM usage in GB (int)
    """
    return int(frames * column_size * nreps * row_size * 8 / 1024**3) + 1
