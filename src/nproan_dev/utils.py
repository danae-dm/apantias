import gc
import os

import numpy as np
from numba import njit, prange

from . import logger
from . import fitting


_logger = logger.Logger(__name__, "info").get_logger()


def get_avg_over_frames(data: np.ndarray) -> np.ndarray:
    """
    Calculates the average over the frames in data.
    Args:
        data: in shape (nframes, column_size, nreps, row_size)

    Returns:
        np.array in shape (column_size, nreps, row_size)
    """
    if np.ndim(data) != 4:
        _logger.error("Input data is not a 4D array.")
        raise ValueError("Input data is not a 4D array.")
    return nanmean(data, axis=0)


def get_avg_over_nreps(data: np.ndarray) -> np.ndarray:
    """
    Calculates the average over the nreps in data.
    Args:
        data: in shape (nframes, column_size, nreps, row_size)
    Returns:
        np.array in shape (nframes, column_size, row_size)
    """
    if np.ndim(data) != 4:
        _logger.error("Input data is not a 4D array.")
        raise ValueError("Input data is not a 4D array.")
    return nanmean(data, axis=2)


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
        data: in shape (nframes, column_size, nreps, row_size)
        avg_over_frames: (optional) in shape (column_size, nreps, row_size)
        avg_over_nreps: (optional) np.in shape (nframes, column_size, row_size)
    Returns:
        np.array in shape (column_size, row_size)
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
            return nanmean(avg_over_frames, axis=1)
        else:
            return nanmean(avg_over_nreps, axis=0)
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
        data: in 1 dimension
        window_size:
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
    """
    return int(frames * column_size * nreps * row_size * 8 / 1024**3) + 1


@njit(parallel=True)
def apply_slope_fit_along_frames(data):
    """
    The equivalent to np.apply_along_axis(func, axis=2, data).
    Args:
        data: 4D np.array
    Returns:
        3D np.array (frame,row,col) with slope value
    """
    if data.ndim != 4:
        _logger.error("Input data is not a 4D array.")
        raise ValueError("Input data is not a 4D array.")
    axis_0_size = data.shape[0]
    axis_1_size = data.shape[1]
    axis_2_size = data.shape[2]
    axis_3_size = data.shape[3]
    output = np.empty((axis_0_size, axis_1_size, axis_3_size))
    for frame in prange(axis_0_size):
        for row in range(axis_1_size):
            for col in range(axis_3_size):
                slope = fitting.linear_fit(data[frame, row, :, col])
                output[frame][row][col] = slope[0]
    return output


def nanmedian(data: np.ndarray, axis: int, keepdims: bool = False) -> np.ndarray:
    """
    The equivalent to np.nanmedian(data, axis=axis, keepdims=keepdims).
    """
    if data.ndim == 2:
        if axis == 0:
            if keepdims:
                return _nanmedian_2d_axis0(data)[np.newaxis, :]
            else:
                return _nanmedian_2d_axis0(data)
        elif axis == 1:
            if keepdims:
                return _nanmedian_2d_axis1(data)[:, np.newaxis]
            else:
                return _nanmedian_2d_axis1(data)
    elif data.ndim == 3:
        if axis == 0:
            if keepdims:
                return _nanmedian_3d_axis0(data)[np.newaxis, :, :]
            else:
                return _nanmedian_3d_axis0(data)
        elif axis == 1:
            if keepdims:
                return _nanmedian_3d_axis1(data)[:, np.newaxis, :]
            else:
                return _nanmedian_3d_axis1(data)
        elif axis == 2:
            if keepdims:
                return _nanmedian_3d_axis2(data)[:, :, np.newaxis]
            else:
                return _nanmedian_3d_axis2(data)
    elif data.ndim == 4:
        if axis == 0:
            if keepdims:
                return _nanmedian_4d_axis0(data)[np.newaxis, :, :, :]
            else:
                return _nanmedian_4d_axis0(data)
        elif axis == 1:
            if keepdims:
                return _nanmedian_4d_axis1(data)[:, np.newaxis, :, :]
            else:
                return _nanmedian_4d_axis1(data)
        elif axis == 2:
            if keepdims:
                return _nanmedian_4d_axis2(data)[:, :, np.newaxis, :]
            else:
                return _nanmedian_4d_axis2(data)
        elif axis == 3:
            if keepdims:
                return _nanmedian_4d_axis3(data)[:, :, :, np.newaxis]
            else:
                return _nanmedian_4d_axis3(data)
    else:
        _logger.error("Data has wrong dimensions")
        return


def nanmean(data: np.ndarray, axis: int, keepdims: bool = False) -> np.ndarray:
    """
    The equivalent to np.nanmean(data, axis=axis, keepdims=keepdims).
    """
    if data.ndim == 2:
        if axis == 0:
            if keepdims:
                return _nanmean_2d_axis0(data)[np.newaxis, :]
            else:
                return _nanmean_2d_axis0(data)
        elif axis == 1:
            if keepdims:
                return _nanmean_2d_axis1(data)[:, np.newaxis]
            else:
                return _nanmean_2d_axis1(data)
    elif data.ndim == 3:
        if axis == 0:
            if keepdims:
                return _nanmean_3d_axis0(data)[np.newaxis, :, :]
            else:
                return _nanmean_3d_axis0(data)
        elif axis == 1:
            if keepdims:
                return _nanmean_3d_axis1(data)[:, np.newaxis, :]
            else:
                return _nanmean_3d_axis1(data)
        elif axis == 2:
            if keepdims:
                return _nanmean_3d_axis2(data)[:, :, np.newaxis]
            else:
                return _nanmean_3d_axis2(data)
    elif data.ndim == 4:
        if axis == 0:
            if keepdims:
                return _nanmean_4d_axis0(data)[np.newaxis, :, :, :]
            else:
                return _nanmean_4d_axis0(data)
        elif axis == 1:
            if keepdims:
                return _nanmean_4d_axis1(data)[:, np.newaxis, :, :]
            else:
                return _nanmean_4d_axis1(data)
        elif axis == 2:
            if keepdims:
                return _nanmean_4d_axis2(data)[:, :, np.newaxis, :]
            else:
                return _nanmean_4d_axis2(data)
        elif axis == 3:
            if keepdims:
                return _nanmean_4d_axis3(data)[:, :, :, np.newaxis]
            else:
                return _nanmean_4d_axis3(data)
    else:
        _logger.error("Data has wrong dimensions")
        return


@njit(parallel=True)
def _nanmedian_4d_axis0(data: np.ndarray) -> np.ndarray:
    """
    The equivalent to np.nanmedian(data, axis=0, keepdims=False).
    Args:
        data: 4D np.array
    Returns:
        3D np.array
    """
    if data.ndim != 4:
        _logger.error("Input data is not a 4D array.")
        raise ValueError("Input data is not a 4D array.")
    axis_1_size = data.shape[1]
    axis_2_size = data.shape[2]
    axis_3_size = data.shape[3]
    output = np.zeros((axis_1_size, axis_2_size, axis_3_size))
    for i in prange(axis_1_size):
        for j in prange(axis_2_size):
            for k in prange(axis_3_size):
                median = np.nanmedian(data[:, i, j, k])
                output[i, j, k] = median
    return output


@njit(parallel=True)
def _nanmedian_4d_axis1(data: np.ndarray) -> np.ndarray:
    """
    The equivalent to np.nanmedian(data, axis=1, keepdims=False).
    Args:
        data: 4D np.array
    Returns:
        3D np.array
    """
    if data.ndim != 4:
        _logger.error("Input data is not a 4D array.")
        raise ValueError("Input data is not a 4D array.")
    axis_0_size = data.shape[0]
    axis_2_size = data.shape[2]
    axis_3_size = data.shape[3]
    output = np.zeros((axis_0_size, axis_2_size, axis_3_size))
    for i in prange(axis_0_size):
        for j in prange(axis_2_size):
            for k in prange(axis_3_size):
                median = np.nanmedian(data[i, :, j, k])
                output[i, j, k] = median
    return output


@njit(parallel=True)
def _nanmedian_4d_axis2(data: np.ndarray) -> np.ndarray:
    """
    The equivalent to np.nanmedian(data, axis=2, keepdims=False).
    Args:
        data: 4D np.array
    Returns:
        3D np.array
    """
    if data.ndim != 4:
        _logger.error("Input data is not a 4D array.")
        raise ValueError("Input data is not a 4D array.")
    axis_0_size = data.shape[0]
    axis_1_size = data.shape[1]
    axis_3_size = data.shape[3]
    output = np.zeros((axis_0_size, axis_1_size, axis_3_size))
    for i in prange(axis_0_size):
        for j in prange(axis_1_size):
            for k in prange(axis_3_size):
                median = np.nanmedian(data[i, j, :, k])
                output[i, j, k] = median
    return output


@njit(parallel=True)
def _nanmedian_4d_axis3(data: np.ndarray) -> np.ndarray:
    """
    The equivalent to np.nanmedian(data, axis=3, keepdims=False).
    Args:
        data: 4D np.array
    Returns:
        3D np.array
    """
    if data.ndim != 4:
        _logger.error("Input data is not a 4D array.")
        raise ValueError("Input data is not a 4D array.")
    axis_0_size = data.shape[0]
    axis_1_size = data.shape[1]
    axis_2_size = data.shape[2]
    output = np.zeros((axis_0_size, axis_1_size, axis_2_size))
    for i in prange(axis_0_size):
        for j in prange(axis_1_size):
            for k in prange(axis_2_size):
                median = np.nanmedian(data[i, j, k, :])
                output[i, j, k] = median
    return output


@njit(parallel=True)
def _nanmedian_3d_axis0(data: np.ndarray) -> np.ndarray:
    """
    The equivalent to np.nanmedian(data, axis=0, keepdims=False).
    Args:
        data: 3D np.array
    Returns:
        2D np.array
    """
    if data.ndim != 3:
        _logger.error("Input data is not a 3D array.")
        raise ValueError("Input data is not a 3D array.")
    axis_1_size = data.shape[1]
    axis_2_size = data.shape[2]
    output = np.zeros((axis_1_size, axis_2_size))
    for i in prange(axis_1_size):
        for j in prange(axis_2_size):
            median = np.nanmedian(data[:, i, j])
            output[i, j] = median
    return output


@njit(parallel=True)
def _nanmedian_3d_axis1(data: np.ndarray) -> np.ndarray:
    """
    The equivalent to np.nanmedian(data, axis=1, keepdims=False).
    Args:
        data: 3D np.array
    Returns:
        2D np.array
    """
    if data.ndim != 3:
        _logger.error("Input data is not a 3D array.")
        raise ValueError("Input data is not a 3D array.")
    axis_0_size = data.shape[0]
    axis_2_size = data.shape[2]
    output = np.zeros((axis_0_size, axis_2_size))
    for i in prange(axis_0_size):
        for j in prange(axis_2_size):
            median = np.nanmedian(data[i, :, j])
            output[i, j] = median
    return output


@njit(parallel=True)
def _nanmedian_3d_axis2(data: np.ndarray) -> np.ndarray:
    """
    The equivalent to np.nanmedian(data, axis=2, keepdims=False).
    Args:
        data: 3D np.array
    Returns:
        2D np.array
    """
    if data.ndim != 3:
        _logger.error("Input data is not a 3D array.")
        raise ValueError("Input data is not a 3D array.")
    axis_0_size = data.shape[0]
    axis_1_size = data.shape[1]
    output = np.zeros((axis_0_size, axis_1_size))
    for i in prange(axis_0_size):
        for j in prange(axis_1_size):
            median = np.nanmedian(data[i, j, :])
            output[i, j] = median
    return output


@njit(parallel=True)
def _nanmedian_2d_axis0(data: np.ndarray) -> np.ndarray:
    """
    The equivalent to np.nanmedian(data, axis=0, keepdims=False).
    Args:
        data: 2D np.array
    Returns:
        1D np.array
    """
    if data.ndim != 2:
        _logger.error("Input data is not a 2D array.")
        raise ValueError("Input data is not a 2D array.")
    axis_1_size = data.shape[1]
    output = np.zeros(axis_1_size)
    for i in prange(axis_1_size):
        median = np.nanmedian(data[:, i])
        output[i] = median
    return output


@njit(parallel=True)
def _nanmedian_2d_axis1(data: np.ndarray) -> np.ndarray:
    """
    The equivalent to np.nanmedian(data, axis=1, keepdims=False).
    Args:
        data: 2D np.array
    Returns:
        1D np.array
    """
    if data.ndim != 2:
        _logger.error("Input data is not a 2D array.")
        raise ValueError("Input data is not a 2D array.")
    axis_0_size = data.shape[0]
    output = np.zeros(axis_0_size)
    for i in prange(axis_0_size):
        median = np.nanmedian(data[i, :])
        output[i] = median
    return output


@njit(parallel=True)
def _nanmean_4d_axis0(data: np.ndarray) -> np.ndarray:
    """
    The equivalent to np.nanmean(data, axis=0, keepdims=False).
    Args:
        data: 4D np.array
    Returns:
        3D np.array
    """
    if data.ndim != 4:
        _logger.error("Input data is not a 4D array.")
        raise ValueError("Input data is not a 4D array.")
    axis_1_size = data.shape[1]
    axis_2_size = data.shape[2]
    axis_3_size = data.shape[3]
    output = np.zeros((axis_1_size, axis_2_size, axis_3_size))
    for i in prange(axis_1_size):
        for j in prange(axis_2_size):
            for k in prange(axis_3_size):
                median = np.nanmean(data[:, i, j, k])
                output[i, j, k] = median
    return output


@njit(parallel=True)
def _nanmean_4d_axis1(data: np.ndarray) -> np.ndarray:
    """
    The equivalent to np.nanmean(data, axis=1, keepdims=False).
    Args:
        data: 4D np.array
    Returns:
        3D np.array
    """
    if data.ndim != 4:
        _logger.error("Input data is not a 4D array.")
        raise ValueError("Input data is not a 4D array.")
    axis_0_size = data.shape[0]
    axis_2_size = data.shape[2]
    axis_3_size = data.shape[3]
    output = np.zeros((axis_0_size, axis_2_size, axis_3_size))
    for i in prange(axis_0_size):
        for j in prange(axis_2_size):
            for k in prange(axis_3_size):
                median = np.nanmean(data[i, :, j, k])
                output[i, j, k] = median
    return output


@njit(parallel=True)
def _nanmean_4d_axis2(data: np.ndarray) -> np.ndarray:
    """
    The equivalent to np.nanmean(data, axis=2, keepdims=False).
    Args:
        data: 4D np.array
    Returns:
        3D np.array
    """
    if data.ndim != 4:
        _logger.error("Input data is not a 4D array.")
        raise ValueError("Input data is not a 4D array.")
    axis_0_size = data.shape[0]
    axis_1_size = data.shape[1]
    axis_3_size = data.shape[3]
    output = np.zeros((axis_0_size, axis_1_size, axis_3_size))
    for i in prange(axis_0_size):
        for j in prange(axis_1_size):
            for k in prange(axis_3_size):
                median = np.nanmean(data[i, j, :, k])
                output[i, j, k] = median
    return output


@njit(parallel=True)
def _nanmean_4d_axis3(data: np.ndarray) -> np.ndarray:
    """
    The equivalent to np.nanmean(data, axis=3, keepdims=False).
    Args:
        data: 4D np.array
    Returns:
        3D np.array
    """
    if data.ndim != 4:
        _logger.error("Input data is not a 4D array.")
        raise ValueError("Input data is not a 4D array.")
    axis_0_size = data.shape[0]
    axis_1_size = data.shape[1]
    axis_2_size = data.shape[2]
    output = np.zeros((axis_0_size, axis_1_size, axis_2_size))
    for i in prange(axis_0_size):
        for j in prange(axis_1_size):
            for k in prange(axis_2_size):
                median = np.nanmean(data[i, j, k, :])
                output[i, j, k] = median
    return output


@njit(parallel=True)
def _nanmean_3d_axis0(data: np.ndarray) -> np.ndarray:
    """
    The equivalent to np.nanmean(data, axis=0, keepdims=False).
    Args:
        data: 3D np.array
    Returns:
        2D np.array
    """
    if data.ndim != 3:
        _logger.error("Input data is not a 3D array.")
        raise ValueError("Input data is not a 3D array.")
    axis_1_size = data.shape[1]
    axis_2_size = data.shape[2]
    output = np.zeros((axis_1_size, axis_2_size))
    for i in prange(axis_1_size):
        for j in prange(axis_2_size):
            median = np.nanmean(data[:, i, j])
            output[i, j] = median
    return output


@njit(parallel=True)
def _nanmean_3d_axis1(data: np.ndarray) -> np.ndarray:
    """
    The equivalent to np.nanmean(data, axis=1, keepdims=False).
    Args:
        data: 3D np.array
    Returns:
        2D np.array
    """
    if data.ndim != 3:
        _logger.error("Input data is not a 3D array.")
        raise ValueError("Input data is not a 3D array.")
    axis_0_size = data.shape[0]
    axis_2_size = data.shape[2]
    output = np.zeros((axis_0_size, axis_2_size))
    for i in prange(axis_0_size):
        for j in prange(axis_2_size):
            median = np.nanmean(data[i, :, j])
            output[i, j] = median
    return output


@njit(parallel=True)
def _nanmean_3d_axis2(data: np.ndarray) -> np.ndarray:
    """
    The equivalent to np.nanmean(data, axis=2, keepdims=False).
    Args:
        data: 3D np.array
    Returns:
        2D np.array
    """
    if data.ndim != 3:
        _logger.error("Input data is not a 3D array.")
        raise ValueError("Input data is not a 3D array.")
    axis_0_size = data.shape[0]
    axis_1_size = data.shape[1]
    output = np.zeros((axis_0_size, axis_1_size))
    for i in prange(axis_0_size):
        for j in prange(axis_1_size):
            median = np.nanmean(data[i, j, :])
            output[i, j] = median
    return output


@njit(parallel=True)
def _nanmean_2d_axis0(data: np.ndarray) -> np.ndarray:
    """
    The equivalent to np.nanmean(data, axis=0, keepdims=False).
    Args:
        data: 2D np.array
    Returns:
        1D np.array
    """
    if data.ndim != 2:
        _logger.error("Input data is not a 2D array.")
        raise ValueError("Input data is not a 2D array.")
    axis_1_size = data.shape[1]
    output = np.zeros(axis_1_size)
    for i in prange(axis_1_size):
        median = np.nanmean(data[:, i])
        output[i] = median
    return output


@njit(parallel=True)
def _nanmean_2d_axis1(data: np.ndarray) -> np.ndarray:
    """
    The equivalent to np.nanmean(data, axis=1, keepdims=False).
    Args:
        data: 2D np.array
    Returns:
        1D np.array
    """
    if data.ndim != 2:
        _logger.error("Input data is not a 2D array.")
        raise ValueError("Input data is not a 2D array.")
    axis_0_size = data.shape[0]
    output = np.zeros(axis_0_size)
    for i in prange(axis_0_size):
        median = np.nanmean(data[i, :])
        output[i] = median
    return output
